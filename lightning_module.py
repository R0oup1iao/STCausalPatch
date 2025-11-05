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
    2.  **消除缓冲区**: 使用预分配张量 (self.epoch_data_pred) 替代 .append() 和 torch.cat()。
    3.  **移除 _TBWrapper**: 在 _log_plots_and_metrics 中直接使用 sklearn。
    4.  **模型切换**: __init__ 默认加载 CUTS_Plus_Transformer_Net, 可手动切换。
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

        # -----------------------------------------------------------------
        # --- 关键修改：手动模型切换 ---
        # -----------------------------------------------------------------
        # 2. 初始化模型和损失
        
        # 从配置中获取模型参数
        pred_config = self.config.data_pred
        
        # 通用模型参数
        model_params = {
            "n_nodes": self.config.n_nodes,
            "in_ch": self.config.data_dim,
            "hidden_ch": pred_config.mlp_hid,
            "shared_weights_decoder": pred_config.shared_weights_decoder,
        }

        # --- 默认加载 Transformer (新架构) ---
        # print(f"Initializing CUTS_Plus_Transformer_Net (Hidden: {pred_config.mlp_hid})")
        # self.model = CUTS_Plus_Transformer_Net(
        #     **model_params,
        #     n_heads=getattr(pred_config, 'n_heads', 4),
        #     transformer_layers=getattr(pred_config, 'transformer_layers', 2),
        #     dropout=getattr(pred_config, 'dropout', 0.0)
        # )
        
        # --- 如需切换回原版, 请注释掉上面, 并取消注释下面这行 ---
        print(f"Initializing CUTS_Plus_Net (Original) (Hidden: {pred_config.mlp_hid})")
        self.model = CUTS_Plus_Net(
            **model_params,
            n_layers=pred_config.gru_layers,
            concat_h=pred_config.concat_h
        )
        # -----------------------------------------------------------------

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
        self.automatic_optimization = False

        # 5. 用于 fill_policy 的预分配张量 (替代缓冲区)
        self.epoch_data_pred = None
        self.epoch_data_pred_all = None
        self.data_interp_buffer = None

    # --------------------------------------------------------------------------
    # 辅助函数 (初始化)
    # --------------------------------------------------------------------------

    def _initialize_graph_params(self):
        """ (重)初始化群组矩阵 (G) 和 因果 logits (GT)。 """
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
        """ (重)创建 Graph Optimizer，并快进到指定 epoch。 """
        total_epochs = self.config.total_epoch
        decay_epochs = total_epochs - self.warmup_epochs
        if decay_epochs <= 0: decay_epochs = 1 # 避免除以零

        gamma = (self.config.graph_discov.lr_graph_end / self.config.graph_discov.lr_graph_start) ** (1 / decay_epochs)
        
        self.opt_graph = torch.optim.Adam([self.causal_logits], lr=self.config.graph_discov.lr_graph_start)
        
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            self.opt_graph, start_factor=1e-6, end_factor=1.0, total_iters=self.warmup_epochs
        )
        decay_sched = torch.optim.lr_scheduler.StepLR(
            self.opt_graph, step_size=1, gamma=gamma
        )
        
        self.sched_graph = torch.optim.lr_scheduler.SequentialLR(
            self.opt_graph, schedulers=[warmup_sched, decay_sched], milestones=[self.warmup_epochs]
        )
        
        current_lr = self.sched_graph.get_last_lr()[0]
        if start_epoch > 0:
            for _ in range(start_epoch):
                self.sched_graph.step()
            current_lr = self.sched_graph.get_last_lr()[0]
        
        print(f"Epoch {start_epoch}: Initialized/Updated graph optimizer. n_groups={self.current_n_groups}, graph_lr={current_lr:.2e}")

    # --------------------------------------------------------------------------
    # 优化器配置
    # --------------------------------------------------------------------------

    def configure_optimizers(self):
        opt_model = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.data_pred.lr_data_start,
            weight_decay=self.config.data_pred.weight_decay
        )
        
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
        
        self.sched_model = torch.optim.lr_scheduler.SequentialLR(
            opt_model, schedulers=[warmup_sched, decay_sched], milestones=[self.warmup_epochs]
        )
        
        return [opt_model]

    # --------------------------------------------------------------------------
    # 辅助函数 (因果图)
    # --------------------------------------------------------------------------

    def get_causal_graph_prob(self):
        """ 辅助函数：计算最终的因果图概率 (原 Graph) """
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
        data_shape = dm.train_data.shape
        self.epoch_data_pred = torch.zeros_like(dm.train_data, device='cpu', dtype=torch.float32)
        self.epoch_data_pred_all = torch.zeros_like(dm.train_data, device='cpu', dtype=torch.float32)
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
        
        graph_prob_detached = self.get_causal_graph_prob().detach()
        graph_sampled_pred = torch.bernoulli(graph_prob_detached.unsqueeze(0).expand(bs, -1, -1)).float()
        
        # --- 接口更新 ---
        # 两个模型现在都接受 (x, mask_x, graph)
        y_pred = self.model(x, mask_x, graph_sampled_pred)
        # --------------
        
        loss_pred = self.pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(mask_y)
        
        self.manual_backward(loss_pred)
        opt_model.step()
        
        self.log("train/pred_loss", loss_pred, on_step=True, on_epoch=True, prog_bar=True)

        return y_pred.detach(), y.detach(), t.detach()

    def _graph_discovery_step(self, batch):
        """ 步骤2：图发现 (训练图) """
        opt_graph = self.opt_graph
        opt_graph.zero_grad()
        
        x, y, t, mask_x, mask_y = batch
        bs = x.shape[0]

        graph_prob = self.get_causal_graph_prob() 
        graph_sampled_disc = self.gumbel_sigmoid_sample(graph_prob, bs, self.current_tau)
        
        # --- 接口更新 ---
        # 两个模型现在都接受 (x, mask_x, graph)
        y_pred_disc = self.model(x, mask_x, graph_sampled_disc)
        # --------------
        
        loss_data = self.pred_loss(y * mask_y, y_pred_disc * mask_y) / torch.mean(mask_y)
        loss_sparsity = torch.linalg.norm(graph_prob.flatten(), ord=1) / (graph_prob.shape[0] * graph_prob.shape[1])
        
        loss_graph = loss_data + self.current_lambda_s * loss_sparsity
        
        self.manual_backward(loss_graph)
        self.opt_graph.step()

        self.log("train/graph_data_loss", loss_data, on_step=True, on_epoch=True)
        self.log("train/graph_sparsity_loss", loss_sparsity, on_step=True, on_epoch=True)
        self.log("train/graph_total_loss", loss_graph, on_step=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y, t, mask_x, mask_y = batch
        
        # --- 步骤 1: 训练模型 ---
        y_pred, y_true, t_indices = self._model_prediction_step(batch)
        
        # --- 步骤 2: 训练图 ---
        self._graph_discovery_step(batch)

        # --- 步骤 3: 写入预分配张量 (替代缓冲区) ---
        if y_pred.shape[2] != 1:
                 print(f"Warning: pred_step is {y_pred.shape[2]}, fill_policy might be incorrect.")

        y_filled = y_pred * (1 - mask_y) + y_true * mask_y
        
        y_filled_squeezed = y_filled.squeeze(2).cpu().float()
        y_pred_squeezed = y_pred.squeeze(2).cpu().float()
        t_cpu = t_indices.cpu()

        self.epoch_data_pred[t_cpu] = y_filled_squeezed
        self.epoch_data_pred_all[t_cpu] = y_pred_squeezed

    # --------------------------------------------------------------------------
    # 训练钩子 (Epoch End)
    # --------------------------------------------------------------------------

    def _update_schedulers_and_annealing(self):
        """ 辅助函数: 更新所有调度器和退火参数 """
        sched_model = self.sched_model

        sched_model.step()
        self.sched_graph.step()

        self.current_tau *= self.gumbel_tau_gamma
        self.current_lambda_s *= self.lambda_gamma

        self.log("params/tau", self.current_tau, on_step=False, on_epoch=True)
        self.log("params/lambda_s", self.current_lambda_s, on_step=False, on_epoch=True)
        self.log("params/lr_model", sched_model.get_last_lr()[0], on_step=False, on_epoch=True)
        self.log("params/lr_graph", self.sched_graph.get_last_lr()[0], on_step=False, on_epoch=True)
        self.log("params/n_groups", self.current_n_groups, on_step=False, on_epoch=True)

    def _apply_fill_policy_and_log_mse(self):
        """ 辅助函数: 应用 fill_policy 并记录 MSE """
        dm = self.trainer.datamodule
        
        dm.update_data(self.epoch_data_pred, self.current_epoch)
        
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
        
        logger = self.logger.experiment
        dm = self.trainer.datamodule
        
        Graph_np = self.get_causal_graph_prob().detach().cpu().numpy()
        true_cm = dm.true_cm
            
        # 1. 绘制时序图
        avg_mask = dm.observ_mask.cpu().numpy().mean(axis=(0, 2))
        time_series_idx = np.argmin(avg_mask) if np.min(avg_mask) < 1 else 0

        # --- 使用你提供的、已修复的绘图代码 ---
        try:
            original = dm.original_data.cpu()[-100:, time_series_idx]
            interp = self.data_interp_buffer.cpu()[-100:, time_series_idx]
            pred = self.epoch_data_pred_all.cpu()[-100:, time_series_idx]

            fig_ts, ax = plt.subplots(figsize=(15, 5))
            ax.plot(original, label='Original Data', color='blue')
            ax.plot(interp, label='Interpolated (Input)', color='orange', linestyle='--')
            ax.plot(pred, label='Model Prediction (y_pred)', color='green', linestyle=':')
            ax.set_title(f"Time Series Comparison (Node {time_series_idx})")
            ax.set_xlabel("Time Step (last 100)")
            ax.set_ylabel("Value")
            ax.legend()
            logger.add_figure("Time Series Comparison", fig_ts, global_step=self.current_epoch)
        except Exception as e:
            print(f"Warning: Failed to log time series plot. {e}")
            plt.close('all')
        # -----------------------------------

        # 2. 绘制因果图矩阵
        n = Graph_np.shape[0]
        figsize = [max(1.5 * n, 10), max(1.0 * n, 10)]
        
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
                plt.close('all')

        # 3. 计算和绘制 AUC/ROC
        if true_cm is not None:
            Graph_transposed = rearrange(Graph_np, "n m -> m n")
            
            try:
                auc = roc_auc_score(true_cm.flatten(), Graph_transposed.flatten())
                self.log("auc", auc, on_step=False, on_epoch=True, prog_bar=True)

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
                plt.close('all')

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
        self._update_schedulers_and_annealing()
        self._apply_fill_policy_and_log_mse()
        
        if (self.current_epoch + 1) % self.config.show_graph_every == 0:
            self._log_plots_and_metrics()

    # --------------------------------------------------------------------------
    # 训练钩子 (Fit End)
    # --------------------------------------------------------------------------

    def on_fit_end(self):
        """ 在训练结束时导出最终因果图。"""
        try:
            graph_np = self.get_causal_graph_prob().detach().cpu().numpy()
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