import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
import numpy as np
from einops import rearrange
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# 假设模型和工具函数在同级 models/ 和 utils/ 目录中
try:
    from .models.cuts_plus_net import CUTS_Plus_Net, CUTS_Plus_Transformer_Net
    from .utils.gumbel_softmax import gumbel_softmax
    from .utils.misc import calc_and_log_metrics, log_time_series, plot_causal_matrix
except ImportError:
    # 允许作为脚本直接运行时导入
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    from models.cuts_plus_net import CUTS_Plus_Net, CUTS_Plus_Transformer_Net
    from utils.gumbel_softmax import gumbel_softmax
    from utils.misc import calc_and_log_metrics, log_time_series, plot_causal_matrix


class CUTSPlusLightning(pl.LightningModule):
    """
    重构后的 CUTS+ Lightning 模块。
    
    核心改动:
    1.  **LR Warmup**: 为两个优化器添加了 10-epoch 线性 warmup。
    2.  **消除缓冲区**: 使用预分配张量 (self.epoch_data_pred) 替代 .append() 和 torch.cat()，
        极大简化了 fill_policy 逻辑。
    3.  **责任分离**: 
        - `training_step` 拆分为 `_model_prediction_step` 和 `_graph_discovery_step`。
        - `on_train_epoch_end` 拆分为 `_update_schedulers_and_annealing`, 
          `_apply_fill_policy_and_log_mse`, 和 `_log_plots_and_metrics`。
    4.  **移除 _TBWrapper**: 在 `_log_plots_and_metrics` 中直接使用 sklearn 计算 AUC 和绘制 ROC，
        移除了丑陋的内联包装类。
    """
    def __init__(self, train_config, reproduc_config):
        super().__init__()
        # 保存配置，自动记录 (lr, batch_size 等)
        self.save_hyperparameters()
        
        self.config = train_config
        self.reproduc_config = reproduc_config
        
        # 1. 重命名变量
        self.group_matrix = None # [N, G], 非训练参数
        self.causal_logits = None # [G, N], 可训练参数
        self.current_n_groups = self.config.n_groups

        # 2. 初始化模型和损失
        self.model = CUTS_Plus_Net(
            n_nodes=self.config.n_nodes,
            in_ch=self.config.data_dim,
            n_layers=self.config.data_pred.gru_layers, 
            hidden_ch=self.config.data_pred.mlp_hid,
            shared_weights_decoder=self.config.data_pred.shared_weights_decoder,
            concat_h=self.config.data_pred.concat_h,
        )
        self.pred_loss = nn.MSELoss()

        # 3. 初始化 Annealing 参数
        self.warmup_epochs = 10 # 新增：Warmup epoch 数量
        end_tau, start_tau = self.config.graph_discov.end_tau, self.config.graph_discov.start_tau
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.config.total_epoch)
        self.current_tau = start_tau
        self.start_tau = start_tau 

        end_lmd, start_lmd = self.config.graph_discov.lambda_s_end, self.config.graph_discov.lambda_s_start
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.config.total_epoch)
        self.current_lambda_s = start_lmd

        # 4. 设置为手动优化
        # 因为 group_policy 需要重置 graph_optimizer，所以必须手动
        self.automatic_optimization = False

        # 5. 用于 fill_policy 的预分配张量 (替代缓冲区)
        # 我们将在 on_train_epoch_start 中初始化它们
        self.epoch_data_pred = None # 存储 y_filled
        self.epoch_data_pred_all = None # 存储 y_pred (用于日志)
        self.data_interp_buffer = None # 存储插值前的数据 (用于日志)

    # --------------------------------------------------------------------------
    # 辅助函数 (初始化)
    # --------------------------------------------------------------------------

    def _initialize_graph_params(self):
        """
        (重)初始化群组矩阵 (G) 和 因果 logits (GT)。
        不再负责创建优化器。
        """
        n_nodes = self.config.n_nodes
        
        # 1. 创建 group_matrix (原 G)
        G = torch.zeros([n_nodes, self.current_n_groups], device=self.device)
        nodes_per_group = n_nodes // self.current_n_groups
        
        i, j = 0, 0
        for i in range(0, self.current_n_groups):
            for j in range(0, nodes_per_group):
                if i * nodes_per_group + j < n_nodes:
                    G[i * nodes_per_group + j, i] = 1
        # 将剩余节点分配给最后一个组
        for k in range(i * nodes_per_group + j + 1, n_nodes):
            G[k, i] = 1
        self.group_matrix = G

        # 2. 创建 causal_logits (原 GT)
        if self.causal_logits is not None and self.config.group_policy is not None:
            # 动态扩展 group (multiply_X_every_Y)
            group_mul = int(self.config.group_policy.split("_")[1])
            GT_init = torch.sigmoid(self.causal_logits).detach().cpu().repeat_interleave(group_mul, 0)[:self.current_n_groups, :]
            GT_init = 1 - (1 - GT_init)**(1 / group_mul)
        else:
            # 第一次初始化
            GT_init = torch.ones((self.current_n_groups, n_nodes)) * 0.5

        # 确保 causal_logits 注册为可训练参数
        self.causal_logits = nn.Parameter(GT_init.to(self.device))
        
    def _initialize_graph_optimizer(self, start_epoch=0):
        """
        (重)创建 Graph Optimizer，并快进到指定 epoch。
        包含 10-epoch warmup 逻辑。
        """
        total_epochs = self.config.total_epoch
        decay_epochs = total_epochs - self.warmup_epochs
        if decay_epochs <= 0: decay_epochs = 1 # 避免除以零

        # 计算衰减率
        gamma = (self.config.graph_discov.lr_graph_end / self.config.graph_discov.lr_graph_start) ** (1 / decay_epochs)
        
        self.opt_graph = torch.optim.Adam([self.causal_logits], lr=self.config.graph_discov.lr_graph_start)
        
        # 定义 Warmup 和 Decay 调度器
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            self.opt_graph, start_factor=1e-6, end_factor=1.0, total_iters=self.warmup_epochs
        )
        decay_sched = torch.optim.lr_scheduler.StepLR(
            self.opt_graph, step_size=1, gamma=gamma
        )
        
        # 链接两个调度器
        self.sched_graph = torch.optim.lr_scheduler.SequentialLR(
            self.opt_graph, schedulers=[warmup_sched, decay_sched], milestones=[self.warmup_epochs]
        )
        
        # 快进调度器到 start_epoch
        # Note: SequentialLR 的 step() 必须在 optimizer.step() 之后调用，
        # 但这里我们只是为了初始化 LR，所以空 step 即可。
        # (在 on_train_epoch_end 中我们会正确地 step)
        current_lr = self.sched_graph.get_last_lr()[0]
        if start_epoch > 0:
            # Pytorch 2.x+ 推荐在 optimizer.step() 之后 step() scheduler。
            # 为了“快进”，我们模拟这个过程。
            # 但对于 LinearLR/StepLR，只调用 .step() 也能更新 LR。
            for _ in range(start_epoch):
                self.sched_graph.step()
            current_lr = self.sched_graph.get_last_lr()[0]
        
        print(f"Epoch {start_epoch}: Initialized/Updated graph optimizer. n_groups={self.current_n_groups}, graph_lr={current_lr:.2e}")

    # --------------------------------------------------------------------------
    # 优化器配置
    # --------------------------------------------------------------------------

    def configure_optimizers(self):
        # 1. 配置模型优化器 (opt_model)
        opt_model = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.data_pred.lr_data_start,
            weight_decay=self.config.data_pred.weight_decay
        )
        
        # 2. 配置模型调度器 (sched_model)
        if "every" in self.config.fill_policy:
            lr_schedule_length = int(self.config.fill_policy.split("_")[-1])
        else:
            lr_schedule_length = self.config.total_epoch
            
        decay_epochs = lr_schedule_length - self.warmup_epochs
        if decay_epochs <= 0: decay_epochs = 1

        gamma = (self.config.data_pred.lr_data_end / self.config.data_pred.lr_data_start) ** (1 / decay_epochs)

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt_model, start_factor=1e-6, end_factor=1.0, total_iters=self.warmup_epochs
        )
        decay_sched = torch.optim.lr_scheduler.StepLR(
            opt_model, step_size=1, gamma=gamma
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt_model, schedulers=[warmup_sched, decay_sched], milestones=[self.warmup_epochs]
        )
        
        return [opt_model], [{"scheduler": scheduler, "interval": "epoch"}]

    # --------------------------------------------------------------------------
    # 辅助函数 (因果图)
    # --------------------------------------------------------------------------

    def get_causal_graph_prob(self):
        """ 辅助函数：计算最终的因果图概率 (原 Graph) """
        # G [N, G], GT [G, N] -> Graph [N, N]
        G = self.group_matrix.to(self.device)
        GT_prob = torch.sigmoid(self.causal_logits)
        return torch.einsum("nm,ml->nl", G, GT_prob)

    def gumbel_sigmoid_sample(self, graph_prob, batch_size, tau):
        """ 辅助函数：Gumbel-Sigmoid 采样 """
        prob = graph_prob[None, :, :, None].expand(batch_size, -1, -1, -1) # [B, N, N, 1]
        logits = torch.concat([prob, (1-prob)], axis=-1) # [B, N, N, 2]
        samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0] # [B, N, N]
        return samples

    # --------------------------------------------------------------------------
    # 训练钩子 (Epoch Start)
    # --------------------------------------------------------------------------

    def on_train_epoch_start(self):
        dm = self.trainer.datamodule
        
        # 1. 处理 group_policy
        if self.config.group_policy is not None:
            group_mul = int(self.config.group_policy.split("_")[1])
            group_every = int(self.config.group_policy.split("_")[3])
            
            needs_init = False
            if self.current_epoch == 0:
                needs_init = True
            elif self.current_epoch % group_every == 0 and self.current_n_groups < self.config.n_nodes:
                self.current_n_groups = min(self.current_n_groups * group_mul, self.config.n_nodes)
                needs_init = True

            if needs_init:
                self._initialize_graph_params()
                self._initialize_graph_optimizer(start_epoch=self.current_epoch)
        
        elif self.current_epoch == 0:
            # group_policy == None，但仍需在 epoch 0 初始化
            self._initialize_graph_params()
            self._initialize_graph_optimizer(start_epoch=0)

        # 2. 处理 supervision_policy
        dm.update_supervision(epoch=self.current_epoch)
        if "masked_before" in self.config.supervision_policy:
            masked_before = int(self.config.supervision_policy.split("_")[2])
            if self.current_epoch == masked_before:
                self.current_tau = self.start_tau # 重置 tau
                self.log("params/tau_reset", self.current_tau, on_step=False, on_epoch=True)
                
        # 3. 预分配张量 (替代缓冲区)
        # clone() 很重要，否则会修改原始数据
        data_shape = dm.train_data.shape
        self.epoch_data_pred = torch.zeros_like(dm.train_data, device='cpu', dtype=torch.float32)
        self.epoch_data_pred_all = torch.zeros_like(dm.train_data, device='cpu', dtype=torch.float32)
        
        # 4. 存储用于插值的 data 副本 (用于 epoch end 日志)
        self.data_interp_buffer = dm.train_data.clone().cpu()

    # --------------------------------------------------------------------------
    # 训练钩子 (Step)
    # --------------------------------------------------------------------------

    def _model_prediction_step(self, batch):
        """ 步骤1：数据预测 (训练模型) """
        opt_model = self.optimizers()
        opt_model.zero_grad()
        
        x, y, t, mask_x, mask_y = batch
        bs = x.shape[0]
        
        # 使用 detach 的图进行预测
        graph_prob_detached = self.get_causal_graph_prob().detach()
        graph_sampled_pred = torch.bernoulli(graph_prob_detached.unsqueeze(0).expand(bs, -1, -1)).float()
        
        y_pred = self.model(x, mask_x, graph_sampled_pred)
        
        loss_pred = self.pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(mask_y)
        
        self.manual_backward(loss_pred)
        opt_model.step()
        
        self.log("train/pred_loss", loss_pred, on_step=True, on_epoch=True, prog_bar=True)

        # 返回需要写入缓冲区的值
        return y_pred.detach(), y.detach(), t.detach()

    def _graph_discovery_step(self, batch):
        """ 步骤2：图发现 (训练图) """
        opt_graph = self.opt_graph
        opt_graph.zero_grad()
        
        x, y, t, mask_x, mask_y = batch
        bs = x.shape[0]

        # 使用带梯度的图
        graph_prob = self.get_causal_graph_prob() 
        graph_sampled_disc = self.gumbel_sigmoid_sample(graph_prob, bs, self.current_tau)
        
        y_pred_disc = self.model(x, mask_x, graph_sampled_disc)
        
        loss_data = self.pred_loss(y * mask_y, y_pred_disc * mask_y) / torch.mean(mask_y)
        loss_sparsity = torch.linalg.norm(graph_prob.flatten(), ord=1) / (graph_prob.shape[0] * graph_prob.shape[1])
        
        loss_graph = loss_data + self.current_lambda_s * loss_sparsity
        
        self.manual_backward(loss_graph)
        self.opt_graph.step()

        self.log("train/graph_data_loss", loss_data, on_step=True, on_epoch=True)
        self.log("train/graph_sparsity_loss", loss_sparsity, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/graph_total_loss", loss_graph, on_step=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y, t, mask_x, mask_y = batch
        
        # --- 步骤 1: 训练模型 ---
        y_pred, y_true, t_indices = self._model_prediction_step(batch)
        
        # --- 步骤 2: 训练图 ---
        self._graph_discovery_step(batch)

        # --- 步骤 3: 写入预分配张量 (替代缓冲区) ---
        # 假设 T_pred = 1 (config.pred_step)
        if y_pred.shape[2] != 1:
             print(f"Warning: pred_step is {y_pred.shape[2]}, fill_policy might be incorrect.")

        y_filled = y_pred * (1 - mask_y) + y_true * mask_y
        
        # 确保 squeeze(2) 是安全的
        y_filled_squeezed = y_filled.squeeze(2).cpu().float()
        y_pred_squeezed = y_pred.squeeze(2).cpu().float()
        t_cpu = t_indices.cpu()

        # 直接写入，无需 append
        self.epoch_data_pred[t_cpu] = y_filled_squeezed
        self.epoch_data_pred_all[t_cpu] = y_pred_squeezed

    # --------------------------------------------------------------------------
    # 训练钩子 (Epoch End)
    # --------------------------------------------------------------------------

    def _update_schedulers_and_annealing(self):
        """ 辅助函数: 更新所有调度器和退火参数 """
        # 1. 更新 Schedulers (手动)
        sched_model = self.lr_schedulers()
        sched_model.step()
        self.sched_graph.step()
        
        # 2. 更新 Annealing 参数
        self.current_tau *= self.gumbel_tau_gamma
        self.current_lambda_s *= self.lambda_gamma
        
        # 3. 日志
        self.log("params/tau", self.current_tau, on_step=False, on_epoch=True)
        self.log("params/lambda_s", self.current_lambda_s, on_step=False, on_epoch=True)
        self.log("params/lr_model", sched_model.get_last_lr()[0], on_step=False, on_epoch=True)
        self.log("params/lr_graph", self.sched_graph.get_last_lr()[0], on_step=False, on_epoch=True)
        self.log("params/n_groups", self.current_n_groups, on_step=False, on_epoch=True)

    def _apply_fill_policy_and_log_mse(self):
        """ 辅助函数: 应用 fill_policy 并记录 MSE """
        dm = self.trainer.datamodule
        
        # 1. 将预分配的张量 (已填充) 应用回 DataModule
        # self.epoch_data_pred 已经在 training_step 中被填充完毕
        dm.update_data(self.epoch_data_pred, self.current_epoch)
        
        # 2. 计算 MSE (现在简单多了)
        mse_pred_to_original = self.pred_loss(
            dm.original_data.to(self.device), 
            self.epoch_data_pred.to(self.device)
        )
        mse_interp_to_original = self.pred_loss(
            dm.original_data.to(self.device), 
            self.data_interp_buffer.to(self.device)
        )

        self.log("metrics/mse_pred_to_original", mse_pred_to_original, on_step=False, on_epoch=True)
        self.log("metrics/mse_interp_to_original", mse_interp_to_original, on_step=False, on_epoch=True)

    def _log_plots_and_metrics(self):
        """ 辅助函数: 绘制所有图像并计算 AUC (替代 _TBWrapper) """
        
        # 0. 获取 Logger 和 DataModule
        logger = self.logger.experiment
        dm = self.trainer.datamodule
        
        Graph_np = self.get_causal_graph_prob().detach().cpu().numpy()
        true_cm = dm.true_cm
            
        # 1. 绘制时序图
        avg_mask = dm.observ_mask.cpu().numpy().mean(axis=(0, 2))
        time_series_idx = np.argmin(avg_mask) if np.min(avg_mask) < 1 else 0
        
        try:
            fig_ts = log_time_series(
                dm.original_data.cpu()[-100:, time_series_idx], 
                self.data_interp_buffer.cpu()[-100:, time_series_idx], 
                self.epoch_data_pred_all.cpu()[-100:, time_series_idx]
            )
            logger.add_figure("Time Series Comparison", fig_ts, global_step=self.current_epoch)
        except Exception as e:
            print(f"Warning: Failed to log time series plot. {e}")
            plt.close('all') # 清理 matplotlib 状态

        # 2. 绘制因果图矩阵
        n = Graph_np.shape[0]
        figsize = [max(1.5 * n, 10), max(1.0 * n, 10)] # 保证最小尺寸
        
        matrix_map = {
            "Group Matrix (G)": self.group_matrix.detach().cpu().numpy(),
            "Causal Logits (GT_prob)": torch.sigmoid(self.causal_logits).detach().cpu().numpy(),
            "Final Graph (Graph)": Graph_np,
            "Ground Truth (true_cm)": true_cm
        }

        for name, matrix in matrix_map.items():
            if matrix is None: continue
            try:
                fig = plot_causal_matrix(matrix, figsize=figsize, show_text=False, vmin=0, vmax=1)
                logger.add_figure(name, fig, global_step=self.current_epoch)
            except Exception as e:
                print(f"Warning: Failed to log matrix '{name}'. {e}")
                plt.close('all') # 清理 matplotlib 状态

        # 3. 计算和绘制 AUC/ROC (替代 _TBWrapper 和 calc_and_log_metrics)
        if true_cm is not None:
            # 原版仓库计算 AUC 时转置了 Graph，我们遵循这个逻辑
            # Graph [N, N] (target, source)
            # true_cm [N, N] (source, target)
            Graph_transposed = rearrange(Graph_np, "n m -> m n")
            
            try:
                # 3.1. 计算 AUC
                auc = roc_auc_score(true_cm.flatten(), Graph_transposed.flatten())
                self.log("metrics/auc", auc, on_step=False, on_epoch=True, prog_bar=True)

                # 3.2. 绘制 ROC 曲线
                fpr, tpr, _ = roc_curve(true_cm.flatten(), Graph_transposed.flatten())
                fig_roc, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                logger.add_figure("ROC Curve", fig_roc, global_step=self.current_epoch)
            
            except Exception as e:
                print(f"Warning: Failed to calculate or log AUC/ROC. {e}")
                plt.close('all') # 清理 matplotlib 状态

            # 3.3. 保存 npz (可选，模仿原版 calc_and_log_metrics)
            try:
                save_dir = self.trainer.log_dir
                if save_dir:
                    npz_path = os.path.join(save_dir, f"graph_metrics_{self.current_epoch}.npz")
                    np.savez(npz_path, 
                             pred_graph=Graph_transposed, 
                             true_graph=true_cm,
                             auc=auc)
            except Exception as e:
                print(f"Warning: Failed to save graph metrics npz. {e}")


    def on_train_epoch_end(self):
        # 1. 更新调度器和退火参数
        self._update_schedulers_and_annealing()
        
        # 2. 应用 fill_policy 并记录 MSE
        self._apply_fill_policy_and_log_mse()
        
        # 3. 绘图和计算指标
        if (self.current_epoch + 1) % self.config.show_graph_every == 0:
            self._log_plots_and_metrics()

    # --------------------------------------------------------------------------
    # 训练钩子 (Fit End)
    # --------------------------------------------------------------------------

    def on_fit_end(self):
        """ 在训练结束时导出最终因果图。"""
        try:
            graph_np = self.get_causal_graph_prob().detach().cpu().numpy()
            
            # 使用 self.trainer.log_dir 获取正确的日志目录
            log_dir = self.trainer.log_dir
            
            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                save_path = os.path.join(log_dir, "Graph_final.npy")
                np.save(save_path, graph_np)
                print(f"Saved final causal graph to {save_path}")
            else:
                 np.save("Graph_final.npy", graph_np)
                 print("Warning: Logger log_dir not found. Saved final graph to Graph_final.npy")

        except Exception as e:
            print(f"Warning: failed to save final Graph_final.npy due to: {e}")