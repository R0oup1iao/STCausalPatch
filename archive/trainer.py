# STCP/trainer.py
# (已修复 Bug B: 强制自注意力)
# (已修复 Bug B: 稀疏损失仅计算非对角线)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import os
import random
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

from utils.logging import log_string
from utils.metrics import metric, masked_mae
from utils.gumbel import gumbel_softmax

class STCPTrainer:
    def __init__(self, config, model, partitioner, dataset_pack, device, log_f):
        self.config = config
        self.model = model
        self.partitioner = partitioner
        self.dataset = dataset_pack['data']
        self.stats = dataset_pack['stats']
        self.ground_truth_adj = dataset_pack['ground_truth_adj']
        self.device = device
        self.log = log_f
        
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.best_causal_f1 = 0.0 
        
        self.loss_null_val = config.training.get('loss_null_val', 0.0)
        if isinstance(self.loss_null_val, str) and self.loss_null_val.lower() == 'none':
            self.loss_null_val = None
            
        log_string(log_f, f"Using loss null_val: {self.loss_null_val}")

        # 
        self.data_pred_loss = masked_mae

        # 1. Data Pred 
        self.data_pred_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr_data_start,
            weight_decay=config.training.weight_decay_data
        )
        self.data_pred_scheduler = optim.lr_scheduler.MultiStepLR(
            self.data_pred_optimizer,
            milestones=config.training.lr_data_milestones,
            gamma=config.training.lr_data_gamma
        )

        # 2. Causal Graph 
        self.G = None
        self.GT = None
        self.graph_optimizer = None
        self.graph_scheduler = None
        
        # 3. 
        self.lambda_s = config.causal.lambda_s_start
        self.gumbel_tau = config.causal.tau_start
        
        self.lambda_gamma = (config.causal.lambda_s_end / config.causal.lambda_s_start) ** (1 / config.training.max_epoch)
        self.tau_gamma = (config.causal.tau_end / config.causal.tau_start) ** (1 / config.training.max_epoch)
        
        self._set_seed(config.training.seed)

    def _set_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            log_string(self.log, f"Seed set to {seed}")

    def _init_causal_graph_params(self, epoch):
        # ... (此函数无变化) ...
        n_nodes = self.config.model.n_nodes
        if self.partitioner is not None:
            current_depth = self.partitioner.get_current_depth(epoch, self.config.training.max_epoch)
            new_partitions = self.partitioner.get_partitions(current_depth)
            n_groups = len(new_partitions)
            if self.G is None or n_groups != self.G.shape[1]:
                log_string(self.log, f"Epoch {epoch}: Hierarchy changing. Depth: {current_depth}, Groups: {n_groups}")
                self.G = self.partitioner.partitions_to_group_matrix(new_partitions, self.device)
                if self.GT is None:
                    GT_init = torch.ones((n_groups, n_nodes)) * 0.5
                else:
                    log_string(self.log, "Re-initializing GT matrix.")
                    GT_init = torch.ones((n_groups, n_nodes)) * 0.5
                self.GT = nn.Parameter(GT_init.to(self.device))
                self._reset_graph_optimizer()
        else:
            if self.G is None:
                log_string(self.log, f"Epoch {epoch}: No partitioner. Using flat (N-to-N) discovery.")
                self.G = torch.eye(n_nodes).to(self.device)
                GT_init = torch.ones((n_nodes, n_nodes)) * 0.5
                self.GT = nn.Parameter(GT_init.to(self.device))
                self._reset_graph_optimizer()

    def _reset_graph_optimizer(self):
        # ... (此函数无变化) ...
        self.graph_optimizer = optim.Adam(
            [self.GT], 
            lr=self.config.training.lr_graph_start,
            weight_decay=self.config.training.weight_decay_graph
        )
        self.graph_scheduler = optim.lr_scheduler.MultiStepLR(
            self.graph_optimizer,
            milestones=self.config.training.lr_graph_milestones,
            gamma=self.config.training.lr_graph_gamma
        )

    def _update_annealing_params(self):
        # ... (此函数无变化) ...
        self.lambda_s *= self.lambda_gamma
        self.gumbel_tau *= self.tau_gamma

    def _get_batch(self, X_data, Y_data, TE_data, indices, batch_idx):
        """
        获取一个批次的数据。
        """
        start_idx = batch_idx * self.config.training.batch_size
        end_idx = min(len(indices), (batch_idx + 1) * self.config.training.batch_size)
        batch_indices = indices[start_idx : end_idx]
        
        X = X_data[batch_indices]
        Y = Y_data[batch_indices]
        TE = TE_data[batch_indices]
        
        # 
        X_norm = (X - self.stats['mean']) / self.stats['std']
        Y_norm = (Y - self.stats['mean']) / self.stats['std'] 
        
        X_torch = torch.from_numpy(X_norm).float().to(self.device)
        Y_torch = torch.from_numpy(Y_norm).float().to(self.device) # 
        TE_torch = torch.from_numpy(TE).float().to(self.device)
        
        return X_torch, Y_torch, TE_torch

    def _run_data_pred_step(self, x_norm, y_norm, te):
        """
        执行一步数据预测 (模型拟合)。
        """
        self.model.train()
        self.data_pred_optimizer.zero_grad()
        
        # 
        Graph = torch.einsum("nm,ml->nl", self.G, torch.sigmoid(self.GT)) # (N, N)
        
        #!#!#! 修复 Bug B: 强制自注意力
        N = Graph.shape[0]
        identity = torch.eye(N, device=Graph.device)
        # 
        Graph_for_sampling = (Graph * (1 - identity)) + identity
        
        # 
        graph_sampled = torch.bernoulli(Graph_for_sampling).float().unsqueeze(0).expand(x_norm.shape[0], -1, -1)
            
        y_pred = self.model(x_norm, te, graph_sampled) # (B, Q, N, 1)
        
        # 
        loss = self.data_pred_loss(y_pred, y_norm, self.loss_null_val)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.data_pred_optimizer.step()
        
        return loss.item()

    def _run_graph_discov_step(self, x_norm, y_norm, te):
        """
        执行一步因果图发现。
        """
        self.model.train()
        self.graph_optimizer.zero_grad()
        
        GT_prob_sig = torch.sigmoid(self.GT)
        Graph = torch.einsum("nm,ml->nl", self.G, GT_prob_sig) # (N, N)

        #!#!#! 修复 Bug B: 强制自注意力
        N = Graph.shape[0]
        identity = torch.eye(N, device=Graph.device)
        # 
        Graph_for_sampling = (Graph * (1 - identity)) + identity

        # 
        Graph_logits = torch.stack([Graph_for_sampling, 1.0 - Graph_for_sampling], dim=-1)
        Graph_logits = torch.log(Graph_logits + 1e-20) # (N, N, 2)
        
        Graph_logits_expanded = Graph_logits.unsqueeze(0).expand(x_norm.shape[0], -1, -1, -1)
        
        # 
        graph_sampled = gumbel_softmax(Graph_logits_expanded, temperature=self.gumbel_tau, hard=True)[..., 0]
        
        #!#!#! 修复 Bug B: 稀疏损失只应惩罚非对角线元素
        loss_sparsity = torch.norm(Graph * (1 - identity), 1) / (N * N)
        
        # 
        with torch.no_grad(): # 
             y_pred = self.model(x_norm, te, graph_sampled)
        
        # 
        loss_data = self.data_pred_loss(y_pred, y_norm, self.loss_null_val)
        
        # 
        loss = loss_data + self.lambda_s * loss_sparsity
        
        loss.backward()
        self.graph_optimizer.step()
        
        return loss.item(), loss_data.item(), loss_sparsity.item()
        
    def _evaluate_causal_graph(self):
        # ... (此函数无变化) ...
        if self.ground_truth_adj is None:
            return 0.0, 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            Graph_pred_prob = torch.einsum("nm,ml->nl", self.G, torch.sigmoid(self.GT)).cpu().numpy()
        gt_adj = self.ground_truth_adj
        np.fill_diagonal(gt_adj, 0)
        np.fill_diagonal(Graph_pred_prob, 0)
        gt_flat = gt_adj.flatten()
        pred_flat = Graph_pred_prob.flatten()
        if np.all(gt_flat == 0) or np.all(gt_flat == 1):
            roc_auc = 0.0
            pr_auc = 0.0
        else:
            roc_auc = roc_auc_score(gt_flat, pred_flat)
            precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
            pr_auc = auc(recall, precision)
        threshold = 0.5 
        pred_adj = (Graph_pred_prob > threshold).astype(int)
        f1 = f1_score(gt_flat, pred_adj.flatten())
        return roc_auc, pr_auc, f1

    def train(self):
        log_string(self.log, "======================TRAIN MODE======================")
        
        trainX, trainY, trainXTE = self.dataset['trainX'], self.dataset['trainY'], self.dataset['trainXTE']
        num_train = trainX.shape[0]
        
        for epoch in range(1, self.config.training.max_epoch + 1):
            start_time = time.time()
            
            self._init_causal_graph_params(epoch)
            self._update_annealing_params()
            
            permutation = np.random.permutation(num_train)
            num_batch = math.ceil(num_train / self.config.training.batch_size)
            
            train_l_sum_data, train_l_sum_graph, train_l_sum_sparse = 0.0, 0.0, 0.0
            
            for batch_idx in range(num_batch):
                # 
                x_norm, y_norm, te = self._get_batch(trainX, trainY, trainXTE, permutation, batch_idx)
                
                # 
                loss_data_step = self._run_data_pred_step(x_norm, y_norm, te)
                
                # 
                if torch.isnan(torch.tensor(loss_data_step)):
                    log_string(self.log, f"Epoch {epoch} Batch {batch_idx}: Data loss is NaN. Stopping graph step.")
                    continue # 
                
                loss_graph_step, loss_g_data, loss_g_sparse = self._run_graph_discov_step(x_norm, y_norm, te)
                
                train_l_sum_data += loss_data_step
                train_l_sum_graph += loss_graph_step
                train_l_sum_sparse += loss_g_sparse

            self.data_pred_scheduler.step()
            if self.graph_scheduler:
                self.graph_scheduler.step()
                
            log_string(self.log, 
                f'Epoch {epoch:03d}, Time: {time.time() - start_time:.1f}s | '
                f'Loss_Data: {train_l_sum_data / num_batch:.4f} | '
                f'Loss_Graph: {train_l_sum_graph / num_batch:.4f} | '
                f'Loss_Sparse: {train_l_sum_sparse / num_batch:.4f}'
            )
            
            # 6. 
            val_loss, roc_auc, pr_auc, f1 = self.validate(epoch)
            
            log_string(self.log,
                f'---> VAL @ {epoch}: MAE: {val_loss:.4f} | '
                f'Causal ROC-AUC: {roc_auc:.4f} | Causal PR-AUC: {pr_auc:.4f} | Causal F1: {f1:.4f}'
            )

            # 
            current_metric = f1 if self.ground_truth_adj is not None else val_loss
            best_metric = self.best_causal_f1 if self.ground_truth_adj is not None else self.best_val_loss
            is_best = (current_metric > best_metric) if self.ground_truth_adj is not None else (current_metric < best_metric)

            if is_best:
                self.best_val_loss = val_loss
                self.best_causal_f1 = f1
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.config.model_save_path)
                log_string(self.log, f'---> Best model saved at epoch {epoch}.')

        log_string(self.log, f'Best epoch is: {self.best_epoch}')

    def validate(self, epoch):
        # 
        log_string(self.log, f"--- Validating at epoch {epoch} ---")
        self.model.eval()
        
        valX, valY, valXTE = self.dataset['valX'], self.dataset['valY'], self.dataset['valXTE']
        num_val = valX.shape[0]
        num_batch = math.ceil(num_val / self.config.training.batch_size)
        
        with torch.no_grad():
            Graph = torch.einsum("nm,ml->nl", self.G, torch.sigmoid(self.GT))
            
            #!#!#! 修复 Bug B: 
            N = Graph.shape[0]
            identity = torch.eye(N, device=Graph.device)
            Graph_for_eval = (Graph * (1 - identity)) + identity
            
            pred_all, label_all = [], []

            for batch_idx in range(num_batch):
                # 
                x_norm, y_norm, te = self._get_batch(valX, valY, valXTE, np.arange(num_val), batch_idx)
                
                B, T, N, _ = x_norm.shape
                graph_expanded = Graph_for_eval.unsqueeze(0).expand(B, -1, -1) # 
                
                y_pred_norm = self.model(x_norm, te, graph_expanded) # (B, Q, N, 1)
                
                # 
                pred_denorm = y_pred_norm.cpu().numpy() * self.stats['std'] + self.stats['mean']
                label_denorm = y_norm.cpu().numpy() * self.stats['std'] + self.stats['mean']
                
                pred_all.append(pred_denorm)
                label_all.append(label_denorm) # 
        
        pred = np.concatenate(pred_all, axis = 0)
        label = np.concatenate(label_all, axis = 0)
        
        maes, rmses, mapes = [], [], []
        for i in range(self.config.data.output_len):
            # 
            mae, rmse , mape = metric(pred[:,i,:,0], label[:,i,:,0], self.loss_null_val)
            maes.append(mae)
        
        avg_mae = np.mean(maes)
        
        # 
        roc_auc, pr_auc, f1 = self._evaluate_causal_graph()
        
        return avg_mae, roc_auc, pr_auc, f1

    def test(self):
        log_string(self.log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.config.model_save_path, map_location=self.device))
        self.model.eval()
        
        testX, testY, testXTE = self.dataset['testX'], self.dataset['testY'], self.dataset['testXTE']
        num_test = testX.shape[0]
        num_batch = math.ceil(num_test / self.config.training.batch_size)
        
        with torch.no_grad():
            Graph = torch.einsum("nm,ml->nl", self.G, torch.sigmoid(self.GT))
            
            #!#!#! 修复 Bug B: 
            N = Graph.shape[0]
            identity = torch.eye(N, device=Graph.device)
            Graph_for_eval = (Graph * (1 - identity)) + identity
            
            pred_all, label_all = [], []

            for batch_idx in range(num_batch):
                # 
                x_norm, y_norm, te = self._get_batch(testX, testY, testXTE, np.arange(num_test), batch_idx)
                
                B, T, N, _ = x_norm.shape
                graph_expanded = Graph_for_eval.unsqueeze(0).expand(B, -1, -1) # 
                y_pred_norm = self.model(x_norm, te, graph_expanded)
                
                pred_denorm = y_pred_norm.cpu().numpy() * self.stats['std'] + self.stats['mean']
                label_denorm = y_norm.cpu().numpy() * self.stats['std'] + self.stats['mean']

                pred_all.append(pred_denorm)
                label_all.append(label_denorm)
        
        pred = np.concatenate(pred_all, axis = 0)
        label = np.concatenate(label_all, axis = 0)
        
        log_string(self.log, "--- Test Results (Prediction) ---")
        maes, rmses, mapes = [], [], []
        for i in range(self.config.data.output_len):
            # 
            mae, rmse , mape = metric(pred[:,i,:,0], label[:,i,:,0], self.loss_null_val)
            maes.append(mae); rmses.append(rmse); mapes.append(mape)
            log_string(self.log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        # 
        mae, rmse, mape = metric(pred[...,0], label[...,0], self.loss_null_val)
        log_string(self.log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        log_string(self.log, "--- Test Results (Causal Graph) ---")
        roc_auc, pr_auc, f1 = self._evaluate_causal_graph()
        log_string(self.log, f'Causal ROC-AUC: {roc_auc:.4f} | Causal PR-AUC: {pr_auc:.4f} | Causal F1: {f1:.4f}')