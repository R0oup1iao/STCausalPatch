import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
import numpy as np
from einops import rearrange
import os

# 假设模型和工具函数在同级 models/ 和 utils/ 目录中
try:
    from .models.cuts_plus_net import CUTS_Plus_Net
    from .utils.gumbel_softmax import gumbel_softmax
    from .utils.misc import calc_and_log_metrics, log_time_series, plot_causal_matrix
except ImportError:
    # 允许作为脚本直接运行时导入
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    from models.cuts_plus_net import CUTS_Plus_Net
    from utils.gumbel_softmax import gumbel_softmax
    from utils.misc import calc_and_log_metrics, log_time_series, plot_causal_matrix


class CUTSPlusLightning(pl.LightningModule):
    """
    替换原 cuts_plus.py 中的 MultiCAD 类。
    管理模型、优化器、训练步骤和绘图。
    """
    def __init__(self, train_config, reproduc_config):
        super().__init__()
        # 保存配置，自动记录 (lr, batch_size 等)
        self.save_hyperparameters()
        
        self.config = train_config
        self.reproduc_config = reproduc_config
        
        # 1. 重命名混乱的变量
        # 原 G -> group_matrix (群组身份矩阵, 非训练参数)
        # 原 GT -> causal_logits (群组到节点的因果图, 可训练参数)
        self.group_matrix = None
        self.causal_logits = None # 将被 on_train_epoch_start 初始化为 nn.Parameter
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
        end_tau, start_tau = self.config.graph_discov.end_tau, self.config.graph_discov.start_tau
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.config.total_epoch)
        self.current_tau = start_tau
        self.start_tau = start_tau # 用于 supervision_policy 重置

        end_lmd, start_lmd = self.config.graph_discov.lambda_s_end, self.config.graph_discov.lambda_s_start
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.config.total_epoch)
        self.current_lambda_s = start_lmd

        # 4. 设置为手动优化
        # 因为 group_policy 需要在 epoch 中途重置 graph_optimizer
        self.automatic_optimization = False

        # 5. 用于 fill_policy 的缓冲区
        # 我们需要在 epoch 结束时重建完整的 data_pred 张量
        self.data_pred_buffer = []
        self.data_t_buffer = []
        self.data_pred_all_buffer = [] # 用于日志
        self.data_interp_buffer = None # 用于日志

    def _initialize_graph_params(self, epoch=0):
        """
        (重)初始化群组矩阵和因果 logits。
        原 cuts_plus.py: 226-261行
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
        if self.causal_logits is not None:
            # 动态扩展 group (multiply_X_every_Y)
            # 复制并扩展现有的 logits
            group_mul = int(self.config.group_policy.split("_")[1])
            GT_init = torch.sigmoid(self.causal_logits).detach().cpu().repeat_interleave(group_mul, 0)[:self.current_n_groups, :]
            GT_init = 1 - (1 - GT_init)**(1 / group_mul)
        else:
            # 第一次初始化
            GT_init = torch.ones((self.current_n_groups, n_nodes)) * 0.5

        # 确保 causal_logits 注册为可训练参数
        self.causal_logits = nn.Parameter(GT_init.to(self.device))
        
        # 3. (重)创建 Graph Optimizer (原 set_graph_optimizer)
        gamma = (self.config.graph_discov.lr_graph_end / self.config.graph_discov.lr_graph_start) ** (1 / self.config.total_epoch)
        lr = self.config.graph_discov.lr_graph_start * (gamma ** epoch)
        
        self.opt_graph = torch.optim.Adam([self.causal_logits], lr=lr)
        self.sched_graph = torch.optim.lr_scheduler.StepLR(self.opt_graph, step_size=1, gamma=gamma)
        print(f"Epoch {epoch}: Initialized/Updated graph params. n_groups={self.current_n_groups}, graph_lr={lr:.2e}")


    def configure_optimizers(self):
        # 只配置模型优化器，图优化器将手动管理
        opt_model = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.data_pred.lr_data_start,
            weight_decay=self.config.data_pred.weight_decay
        )
        
        if "every" in self.config.fill_policy:
            lr_schedule_length = int(self.config.fill_policy.split("_")[-1])
        else:
            lr_schedule_length = self.config.total_epoch
            
        gamma = (self.config.data_pred.lr_data_end / self.config.data_pred.lr_data_start) ** (1 / lr_schedule_length)
        sched_model = torch.optim.lr_scheduler.StepLR(opt_model, step_size=1, gamma=gamma)
        
        return [opt_model], [sched_model]

    def get_causal_graph_prob(self):
        """ 辅助函数：计算最终的因果图概率 (原 Graph) """
        # G [N, G], GT [G, N] -> Graph [N, N]
        # 确保两侧张量在同一设备，避免 CPU/CUDA 混用
        device = self.causal_logits.device
        G = self.group_matrix.to(device)
        GT_prob = torch.sigmoid(self.causal_logits)
        return torch.einsum("nm,ml->nl", G, GT_prob)

    def gumbel_sigmoid_sample(self, graph_prob, batch_size, tau):
        """ 辅助函数：Gumbel-Sigmoid 采样 (原 graph_discov 内) """
        prob = graph_prob[None, :, :, None].expand(batch_size, -1, -1, -1) # [B, N, N, 1]
        logits = torch.concat([prob, (1-prob)], axis=-1) # [B, N, N, 2]
        samples = gumbel_softmax(logits, tau=tau, hard=True)[:, :, :, 0] # [B, N, N]
        return samples

    def on_train_epoch_start(self):
        # 1. 处理 group_policy (原 cuts_plus.py: 226-261)
        if self.config.group_policy is not None:
            group_mul = int(self.config.group_policy.split("_")[1])
            group_every = int(self.config.group_policy.split("_")[3])
            
            if self.current_epoch == 0:
                # 第一次初始化
                self._initialize_graph_params(epoch=0)
            elif self.current_epoch % group_every == 0 and self.current_n_groups < self.config.n_nodes:
                self.current_n_groups *= group_mul
                if self.current_n_groups > self.config.n_nodes:
                    self.current_n_groups = self.config.n_nodes
                # 重置图参数和优化器
                self._initialize_graph_params(epoch=self.current_epoch)
        
        elif self.current_epoch == 0:
            # group_policy == None，但仍需在 epoch 0 初始化
            self._initialize_graph_params(epoch=0)

        # 2. 处理 supervision_policy (原 cuts_plus.py: 282-287)
        dm = self.trainer.datamodule
        dm.update_supervision(epoch=self.current_epoch)
        # 如果策略触发，重置 tau
        if "masked_before" in self.config.supervision_policy:
            masked_before = int(self.config.supervision_policy.split("_")[2])
            if self.current_epoch == masked_before:
                self.current_tau = self.start_tau
                self.log("params/tau", self.current_tau, on_step=False, on_epoch=True)
                
        # 3. 清空缓冲区
        self.data_pred_buffer.clear()
        self.data_t_buffer.clear()
        self.data_pred_all_buffer.clear()
        
        # 4. 存储用于插值的 data 副本 (用于 epoch end 日志)
        self.data_interp_buffer = dm.train_data.clone().cpu()


    def training_step(self, batch, batch_idx):
        opt_model = self.optimizers()
        opt_graph = self.opt_graph # 手动管理的优化器
        
        x, y, t, mask_x, mask_y = batch
        bs = x.shape[0]
        
        # --- 1. 数据预测步骤 (原 latent_data_pred) ---
        opt_model.zero_grad()
        
        graph_prob_detached = self.get_causal_graph_prob().detach()
        # 使用 torch.bernoulli (原 sample_bernoulli)
        graph_sampled_pred = torch.bernoulli(graph_prob_detached.unsqueeze(0).expand(bs, -1, -1)).float()
        
        y_pred = self.model(x, mask_x, graph_sampled_pred)
        
        loss_pred = self.pred_loss(y * mask_y, y_pred * mask_y) / torch.mean(mask_y)
        
        self.manual_backward(loss_pred)
        opt_model.step()
        
        self.log("train/pred_loss", loss_pred, on_step=True, on_epoch=True, prog_bar=True)

        # 存储结果用于 fill_policy 和 logging
        y_filled = y_pred * (1 - mask_y) + y * mask_y
        self.data_pred_buffer.append(y_filled.detach().cpu())
        self.data_t_buffer.append(t.cpu())
        self.data_pred_all_buffer.append(y_pred.detach().cpu())

        # --- 2. 图发现步骤 (原 graph_discov) ---
        opt_graph.zero_grad()
        
        graph_prob = self.get_causal_graph_prob() # 保持梯度
        # 使用 gumbel_sigmoid_sample
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


    def on_train_epoch_end(self):
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

        # 4. 处理 fill_policy (原 cuts_plus.py: 263-280)
        dm = self.trainer.datamodule
        
        # 4.1. 重建 data_pred 张量
        # clone() 很重要，否则 data_pred 和 dm.train_data 指向同一个内存
        data_pred = dm.train_data.clone().cpu() 
        all_y_filled = torch.cat(self.data_pred_buffer, dim=0) # [Total, N, T_pred, D]
        all_t = torch.cat(self.data_t_buffer, dim=0)           # [Total]
        
        # 确保 T_pred=1, D=1 (基于原代码的假设)
        if all_y_filled.shape[2] != 1:
             print(f"Warning: pred_step is {all_y_filled.shape[2]}, fill_policy might be incorrect.")
             
        # [Total, N, D]
        all_y_filled_squeezed = all_y_filled.squeeze(2).float() 
        
        # 将 [Total, N, D] 的预测值放回 [T, N, D] 的 data_pred 张量中
        # 注意：这里我们假设 T_pred = 1
        data_pred[all_t] = all_y_filled_squeezed
        
        # 4.2. 将 data_pred 应用回 DataModule
        # 传入 CPU 张量，DataModule 内部会对齐设备
        dm.update_data(data_pred, self.current_epoch)
        
        # 4.3. 计算 MSE (原 cuts_plus.py: 326-330)
        mse_pred_to_original = self.pred_loss(dm.original_data.to(self.device), data_pred.to(self.device))
        mse_interp_to_original = self.pred_loss(dm.original_data.to(self.device), self.data_interp_buffer.to(self.device))

        self.log("metrics/mse_pred_to_original", mse_pred_to_original, on_step=False, on_epoch=True)
        self.log("metrics/mse_interp_to_original", mse_interp_to_original, on_step=False, on_epoch=True)

        # 5. 绘图 (原 cuts_plus.py: 360-386)
        if (self.current_epoch + 1) % self.config.show_graph_every == 0:
            # 5.1. 绘制时序图
            # 重建 data_pred_all 张量
            data_pred_all = dm.train_data.clone().cpu()
            all_y_pred = torch.cat(self.data_pred_all_buffer, dim=0)
            all_y_pred_squeezed = all_y_pred.squeeze(2).float()
            data_pred_all[all_t] = all_y_pred_squeezed
            
            # 选择一个 time_series_idx (原版逻辑)
            avg_mask = dm.observ_mask.cpu().numpy().mean(axis=(0, 2))
            time_series_idx = np.argmin(avg_mask) if np.min(avg_mask) < 1 else 0
            
            # fig_ts = log_time_series(
            #     dm.original_data.cpu()[-100:, time_series_idx], 
            #     self.data_interp_buffer.cpu()[-100:, time_series_idx], 
            #     data_pred_all.cpu()[-100:, time_series_idx]
            # )
            # self.logger.experiment.add_figure("Time Series Comparison", fig_ts, global_step=self.current_epoch)
            
            # 5.2. 绘制因果图
            G_prob_np = self.group_matrix.detach().cpu().numpy()
            GT_prob_np = torch.sigmoid(self.causal_logits).detach().cpu().numpy()
            Graph_np = self.get_causal_graph_prob().detach().cpu().numpy()
            
            # 绘制三个矩阵以及 true_cm
            n = Graph_np.shape[0]
            figsize = [1.5 * n, 1.0 * n]
            try:
                fig_g = plot_causal_matrix(G_prob_np, figsize=figsize, show_text=False, vmin=0, vmax=1)
                self.logger.experiment.add_figure("Group Matrix (G)", fig_g, global_step=self.current_epoch)
            except Exception:
                pass
            try:
                fig_gt = plot_causal_matrix(GT_prob_np, figsize=figsize, show_text=False, vmin=0, vmax=1)
                self.logger.experiment.add_figure("Causal Logits (GT_prob)", fig_gt, global_step=self.current_epoch)
            except Exception:
                pass
            try:
                fig_graph = plot_causal_matrix(Graph_np, figsize=figsize, show_text=False, vmin=0, vmax=1)
                self.logger.experiment.add_figure("Final Graph (Graph)", fig_graph, global_step=self.current_epoch)
            except Exception:
                pass
            try:
                if dm.true_cm is not None:
                    fig_true = plot_causal_matrix(dm.true_cm, figsize=figsize, show_text=False, vmin=0, vmax=1)
                    self.logger.experiment.add_figure("Ground Truth (true_cm)", fig_true, global_step=self.current_epoch)
            except Exception:
                pass

            # 5.3. 计算 AUC (原 cuts_plus.py: 389-391)
            if dm.true_cm is not None:
                Graph_transposed = rearrange(Graph_np, "n m -> m n")

                # 为兼容 utils.misc.calc_and_log_metrics 的日志接口，构建一个轻量包装器
                class _TBWrapper:
                    def __init__(self, tb_logger, base_logger):
                        # 需要 tblogger.add_figure
                        self.tblogger = tb_logger
                        self._base_logger = base_logger
                        # 解析 log_dir 用于保存 npz
                        self._log_dir = None
                        if hasattr(base_logger, "log_dir") and base_logger.log_dir is not None:
                            self._log_dir = base_logger.log_dir
                        elif hasattr(base_logger, "save_dir") and hasattr(base_logger, "name") and hasattr(base_logger, "version"):
                            self._log_dir = os.path.join(base_logger.save_dir, base_logger.name, f"version_{base_logger.version}")
                        if self._log_dir is not None:
                            os.makedirs(self._log_dir, exist_ok=True)

                    def log_metrics(self, metrics_dict, step):
                        # 将字典写入 TensorBoard scalar
                        for k, v in metrics_dict.items():
                            try:
                                scalar = v.item() if hasattr(v, "item") else float(v)
                            except Exception:
                                continue
                            self.tblogger.add_scalar(k, scalar, global_step=step)

                    def log_npz(self, name, data, iters):
                        # 将 npz 存到日志目录
                        if self._log_dir is None:
                            return
                        np.savez(os.path.join(self._log_dir, f"{name}_{iters}.npz"), **data)

                wrapper = _TBWrapper(self.logger.experiment, self.logger)
                auc = calc_and_log_metrics(
                    Graph_transposed,
                    dm.true_cm,
                    wrapper,
                    self.current_epoch,
                    plot_roc=True,
                )
                self.log("metrics/auc", auc, on_step=False, on_epoch=True, prog_bar=True)

    def on_fit_end(self):
        """在训练结束时导出最终因果图，以便复现实验结果。"""
        try:
            graph_np = self.get_causal_graph_prob().detach().cpu().numpy()
            # 保存到 logger 的文件夹
            log_dir = None
            if hasattr(self.logger, "log_dir") and self.logger.log_dir is not None:
                log_dir = self.logger.log_dir
            elif hasattr(self.logger, "save_dir") and hasattr(self.logger, "name"):
                log_dir = os.path.join(self.logger.save_dir, self.logger.name)
            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                np.save(os.path.join(log_dir, "Graph.npy"), graph_np)
        except Exception as e:
            print(f"Warning: failed to save final Graph.npy due to: {e}")