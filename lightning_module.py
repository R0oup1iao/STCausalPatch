import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.cuts_model import CUTSModel
from models.causal_learner import CausalGraphLearner

class CUTSPlusLightning(pl.LightningModule):
    """
    PyTorch Lightning 模块，整合模型和训练逻辑
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg) # 保存所有配置
        
        self.cfg_data = cfg.data
        self.cfg_model = cfg.model
        self.cfg_optim = cfg.optimizer
        self.cfg_causal = cfg.causal
        
        # 1. 预测模型
        self.model = CUTSModel(self.cfg_model, self.cfg_data)
        
        # 2. 因果图学习器
        self.causal_learner = CausalGraphLearner(
            n_nodes=self.cfg_data.get("n_nodes"),
            n_groups=self.cfg_causal.get("n_groups_start"),
            group_policy=self.cfg_causal.get("group_policy"),
            gumbel_tau_start=self.cfg_causal.get("start_tau")
        )
        
        # 3. Loss (与原代码一致)
        self.data_pred_loss = nn.MSELoss(reduction='sum') # 使用 sum，然后手动除以 mask 均值
        
        # 4. 因果发现超参数 (从原代码移植)
        self.lambda_s = self.cfg_causal.get("lambda_s_start")
        self.lambda_gamma = (self.cfg_causal.get("lambda_s_end") / 
                             self.cfg_causal.get("lambda_s_start")) ** (1.0 / self.hparams.trainer.max_epochs)
        
        self.gumbel_tau_gamma = (self.cfg_causal.get("end_tau") / 
                                 self.cfg_causal.get("start_tau")) ** (1.0 / self.hparams.trainer.max_epochs)

        # 标记是否需要重置优化器（当 G 更新时）
        self.needs_optimizer_reset = False
        self.automatic_optimization = False # 我们将手动管理双优化器

    def _calculate_loss(self, y_pred, y, mask_y):
        """
        计算带掩码的 MSE loss，与原代码 (line 180) 完全一致
        """
        loss = self.data_pred_loss(y * mask_y, y_pred * mask_y)
        # 防止除以 0
        mask_mean = torch.mean(mask_y)
        return loss / (mask_mean + 1e-8)

    def training_step(self, batch, batch_idx):
        """
        执行一个训练步骤，包含两个优化器
        """
        opt_model, opt_graph = self.optimizers()
        
        x, y, mask_x, mask_y, t = batch
        batch_size = x.shape[0]
        n_nodes = x.shape[1]
        
        # (N, N)
        effective_adj = self.causal_learner.get_effective_adj_matrix()

        # --- 优化器 0: 数据预测 (latent_data_pred) ---
        # 移植自 'cuts_plus.py' (lines 160-184)
        
        # (B, N, N)
        graph_sampled_pred = self.causal_learner.sample_graph(
            effective_adj.detach(), # 图参数不参与此步优化
            batch_size, 
            mode='bernoulli'
        )
        
        y_pred = self.model(x, mask_x, graph_sampled_pred)
        
        loss_pred = self._calculate_loss(y_pred, y, mask_y)
        
        opt_model.zero_grad()
        self.manual_backward(loss_pred)
        opt_model.step()
        
        self.log('train/pred_loss', loss_pred, on_step=True, on_epoch=True, prog_bar=True)
        
        # --- 优化器 1: 因果图发现 (graph_discov) ---
        # 移植自 'cuts_plus.py' (lines 186-210)
        
        # (B, N, N)
        graph_sampled_discov = self.causal_learner.sample_graph(
            effective_adj, # 图参数参与此步优化
            batch_size, 
            mode='gumbel'
        )
        
        # 稀疏性损失 (line 203)
        loss_sparsity = torch.linalg.norm(effective_adj.flatten(), ord=1) / (n_nodes * n_nodes)
        
        # 数据损失 (line 205)
        # 注意：原代码在此处也计算 y_pred，梯度会流经模型和图
        y_pred_graph = self.model(x, mask_x, graph_sampled_discov)
        loss_data = self._calculate_loss(y_pred_graph, y, mask_y)
        
        # 总损失 (line 207)
        loss_graph_total = loss_sparsity * self.lambda_s + loss_data
        
        opt_graph.zero_grad()
        self.manual_backward(loss_graph_total)
        opt_graph.step()
        
        self.log_dict({
            'train/graph_total_loss': loss_graph_total,
            'train/graph_data_loss': loss_data,
            'train/graph_sparsity_loss': loss_sparsity,
            'params/lambda_s': self.lambda_s
        }, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        """
        在每个 epoch 结束时更新超参数
        """
        # 1. 更新 gumbel tau 和 lambda_s (line 350-351)
        self.causal_learner.gumbel_tau *= self.gumbel_tau_gamma
        self.lambda_s *= self.lambda_gamma
        
        self.log_dict({
            'params/gumbel_tau': self.causal_learner.gumbel_tau,
            'params/lambda_s_next': self.lambda_s
        })
        
        # 2. 检查是否需要更新 G (line 259)
        if self.causal_learner.update_groups(self.current_epoch):
            # 如果 G 更新了，GT (learnable_grouped_adj) 也被替换了
            # 我们需要重置图优化器
            self.needs_optimizer_reset = True

    def configure_optimizers(self):
        """
        配置两个优化器和对应的学习率调度器
        """
        # 优化器 1: 预测模型
        opt_model = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg_optim.data_pred.lr_data_start,
            weight_decay=self.cfg_optim.data_pred.weight_decay
        )
        # 优化器 2: 因果图
        opt_graph = torch.optim.Adam(
            self.causal_learner.parameters(), # 只优化 GT
            lr=self.cfg_optim.graph_discov.lr_graph_start
        )
        
        # 调度器 1
        lr_schedule_length = self.hparams.trainer.max_epochs
        gamma1 = (self.cfg_optim.data_pred.lr_data_end / 
                  self.cfg_optim.data_pred.lr_data_start) ** (1.0 / lr_schedule_length)
        sched1 = torch.optim.lr_scheduler.StepLR(opt_model, step_size=1, gamma=gamma1)
        
        # 调度器 2
        gamma2 = (self.cfg_optim.graph_discov.lr_graph_end / 
                  self.cfg_optim.graph_discov.lr_graph_start) ** (1.0 / lr_schedule_length)
        sched2 = torch.optim.lr_scheduler.StepLR(opt_graph, step_size=1, gamma=gamma2)
        
        return [opt_model, opt_graph], [sched1, sched2]

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """
        在图优化器步骤之前，检查是否需要重置它
        """
        if optimizer_idx == 1 and self.needs_optimizer_reset:
            print(f"Epoch {self.current_epoch}: Resetting graph optimizer due to group update.")
            opt_graph = self.optimizers()[1]
            
            # 重新创建优化器
            new_opt_graph = torch.optim.Adam(
                self.causal_learner.parameters(),
                lr=self.cfg_optim.graph_discov.lr_graph_start # 重置学习率
            )
            opt_graph.load_state_dict(new_opt_graph.state_dict())
            
            # 重新创建调度器
            gamma2 = (self.cfg_optim.graph_discov.lr_graph_end / 
                      self.cfg_optim.graph_discov.lr_graph_start) ** (1.0 / self.hparams.trainer.max_epochs)
            
            # 让调度器从当前 gamma 开始
            # 我们需要手动计算当前 epoch 对应的 gamma
            current_gamma = gamma2 ** self.current_epoch 
            new_sched_graph = torch.optim.lr_scheduler.StepLR(opt_graph, step_size=1, gamma=gamma2)
            # 设置调度器的 last_epoch 来同步
            new_sched_graph.last_epoch = self.current_epoch 
            
            self.lr_schedulers()[1].load_state_dict(new_sched_graph.state_dict())
            
            self.needs_optimizer_reset = False
            
    # 你可以在这里添加 validation_step 来计算 AUC 等指标
    def validation_step(self, batch, batch_idx):
        # 示例：计算验证集上的预测损失
        x, y, mask_x, mask_y, t = batch
        batch_size = x.shape[0]
        
        effective_adj = self.causal_learner.get_effective_adj_matrix()
        graph_sampled_pred = self.causal_learner.sample_graph(
            effective_adj.detach(), 
            batch_size, 
            mode='bernoulli'
        )
        
        y_pred = self.model(x, mask_x, graph_sampled_pred)
        loss_val = self._calculate_loss(y_pred, y, mask_y)
        self.log('val/pred_loss', loss_val, prog_bar=True)
        return loss_val