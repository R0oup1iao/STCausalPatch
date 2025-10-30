import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def _generate_indices(input_step, pred_step, t_length, block_size=None):
    """
    辅助函数：从旧代码 'generate_indices' 移植而来，用于创建所有可能的样本索引。
    """
    if block_size is None:
        block_size = t_length
        
    offsets_in_block = np.arange(input_step, block_size - pred_step + 1)
    assert t_length % block_size == 0, "t_length % block_size != 0"
    random_t_list = []
    for block_start in range(0, t_length, block_size):
        random_t_list += (offsets_in_block + block_start).tolist()
    
    # 注意：在 Dataset 中我们只生成索引列表，打乱(shuffle)的工作交给 DataLoader
    return random_t_list

class TimeSeriesDataset(Dataset):
    """
    用于 CUTS+ 的 PyTorch Dataset
    """
    def __init__(self, data, observ_mask, input_step, pred_step, block_size=None):
        """
        初始化 Dataset
        :param data: 完整的时间序列数据 (T, N, D)
        :param observ_mask: 对应的掩码 (T, N, D)
        :param input_step: 输入序列长度 (S)
        :param pred_step: 预测序列长度 (P)
        :param block_size: (可选) 对应旧代码中的 block_size
        """
        super().__init__()
        self.data = torch.from_numpy(data).float()
        self.observ_mask = torch.from_numpy(observ_mask).float()
        self.input_step = input_step
        self.pred_step = pred_step
        
        t_length = data.shape[0]
        # 生成所有可能的采样起始点（t）
        # t 代表的是 y 的起始点
        self.indices = _generate_indices(input_step, pred_step, t_length, block_size)

    def __len__(self):
        """
        返回数据集中的样本总数
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个样本
        """
        # t 是 y 的起始时间戳
        t = self.indices[idx]
        
        # 提取 x, y, mask_x, mask_y
        # 形状: (T, N, D) -> (N, T, D)
        
        x = self.data[t - self.input_step : t, :].permute(1, 0, 2)
        y = self.data[t : t + self.pred_step, :].permute(1, 0, 2)
        
        mask_x = self.observ_mask[t - self.input_step : t, :].permute(1, 0, 2)
        mask_y = self.observ_mask[t : t + self.pred_step, :].permute(1, 0, 2)
        
        # t 作为元数据也一并返回，用于旧代码中的 data_pred 更新
        return x, y, mask_x, mask_y, t

class TimeSeriesDataModule(pl.LightningDataModule):
    """
    用于 CUTS+ 的 PyTorch Lightning DataModule
    """
    def __init__(self, config: dict):
        """
        简化了 __init__，只接收 data 相关的配置
        """
        super().__init__()
        # 将 data: 下的所有配置保存
        self.cfg = config 
        
        # 从配置中读取参数
        self.input_step = self.cfg.get("input_step")
        self.pred_step = self.cfg.get("pred_step")
        self.batch_size = self.cfg.get("batch_size")
        self.block_size = self.cfg.get("block_size", None)
        self.num_workers = self.cfg.get("num_workers", 4)
        
        self.data = None
        self.mask = None
        self.true_cm = None # 用于存储真实的因果图

    def _preprocess(self, data):
        """
        从旧代码 'prepross_data' (cuts_plus.py, line 375) 移植
        """
        T, N, D = data.shape
        new_data = np.zeros_like(data, dtype=float)
        for i in range(N):
            node = data[:, i, :]
            # 标准化
            mean = np.mean(node[self.mask[:, i, :] == 1]) if np.any(self.mask[:, i, :] == 1) else 0
            std = np.std(node[self.mask[:, i, :] == 1]) if np.any(self.mask[:, i, :] == 1) and np.std(node[self.mask[:, i, :] == 1]) > 1e-6 else 1
            new_data[:, i, :] = (node - mean) / std
        return new_data

    def _generate_mask(self, data_shape, seed, params):
        """
        从你提供的 notebook 脚本中移植的掩码生成逻辑
        """
        print(f"--- 正在生成掩码 (Seed: {seed}) ---")
        np.random.seed(seed)
        rand = np.random.random
        randint = np.random.randint
        
        p_block = params.get('p_block', 0.05)
        p_noise = params.get('p_noise', 0.0)
        max_seq = params.get('max_seq', 10)
        min_seq = params.get('min_seq', 1)

        # 1. 生成随机块噪声 (block noise)
        init_mask = rand(data_shape) < p_block
        for col in range(init_mask.shape[1]): # 遍历 N
            idxs = np.flatnonzero(init_mask[:, col])
            if not len(idxs):
                continue
            
            fault_len = min_seq
            if max_seq > min_seq:
                fault_len_range = max_seq - min_seq
                # 为每个起始点 idxs 生成一个随机长度
                fault_lengths = fault_len + randint(0, fault_len_range + 1, size=len(idxs))
            else:
                fault_lengths = np.full(len(idxs), fault_len)

            # 创建扩展索引
            idxs_ext = []
            for i, start_idx in enumerate(idxs):
                idxs_ext.append(np.arange(start_idx, start_idx + fault_lengths[i]))
            
            if not idxs_ext:
                continue

            idxs_ext = np.concatenate(idxs_ext)
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, init_mask.shape[0] - 1)
            init_mask[idxs, col] = True

        # 2. 生成随机点噪声 (point noise)
        eval_mask = init_mask | (rand(init_mask.shape) < p_noise)
        
        # 3. 反转掩码 (1 = 观测到, 0 = 缺失)
        mask = 1 - eval_mask
        
        print(f"掩码生成完毕. 观测率: {mask.mean() * 100:.2f}%")
        return mask.astype(int)
    def setup(self, stage: str = None):
        """
        加载和预处理数据，创建 Datasets
        """
        # 仅在第一次调用时加载/生成数据
        if self.data is not None:
            return

        if self.cfg.source == 'simulate':
            print(f"--- 模式: 动态模拟数据 ---")
            # 动态导入 simu_data 模块
            # 这可以避免在 'file' 模式下导入它
            from . import simu_data
            
            sim_name = self.cfg.simulation.name
            sim_params = self.cfg.simulation.params
            print(f"调用: {sim_name}(**{sim_params})")
            
            # 获取模拟函数
            sim_func = getattr(simu_data, sim_name, None)
            if sim_func is None:
                raise ImportError(f"在 data/simu_data.py 中未找到函数 {sim_name}")
                
            # (!!!) 调用模拟函数
            # 我们假设它返回 (data, true_cm)
            raw_data, self.true_cm = sim_func(**sim_params)
            
            # 确保数据是 (T, N, D=1)
            if raw_data.ndim == 2: # (T, N)
                raw_data = raw_data[..., None]
            
            # (!!!) 生成掩码
            self.mask = self._generate_mask(
                data_shape=raw_data.shape,
                seed=self.cfg.missing.seed,
                params=self.cfg.missing.params
            )

        elif self.cfg.source == 'file':
            print(f"--- 模式: 从文件加载数据 ---")
            # 假设数据是 .npy 文件
            raw_data = np.load(self.cfg.file.data_path)
            self.mask = np.load(self.cfg.file.mask_path)
            # (真实因果图可以作为可选文件加载)
            # self.true_cm = np.load(...) 
            
            if raw_data.ndim == 2: # (T, N)
                raw_data = raw_data[..., None]
            if self.mask.ndim == 2: # (T, N)
                self.mask = self.mask[..., None]
        
        else:
            raise ValueError(f"未知的 data.source: {self.cfg.source}")

        # (!!!) 预处理（标准化）
        # 这一步对 'file' 和 'simulate' 模式都通用
        print("--- 正在预处理 (标准化) 数据 ---")
        self.data = self._preprocess(raw_data)
        
        # 检查配置是否匹配
        if self.data.shape[1] != self.cfg.n_nodes or self.data.shape[2] != self.cfg.data_dim:
            print(f"警告: 数据形状 {self.data.shape} (T, N, D) 与配置不符")
            print(f"Config n_nodes: {self.cfg.n_nodes}, data N: {self.data.shape[1]}")
            print(f"Config data_dim: {self.cfg.data_dim}, data D: {self.data.shape[2]}")
            # 更新配置以匹配数据
            self.cfg.n_nodes = self.data.shape[1]
            self.cfg.data_dim = self.data.shape[2]

        # 假设我们使用所有数据进行训练（原代码没有划分验证集）
        if stage == 'fit' or stage is None:
            self.train_dataset = TimeSeriesDataset(
                data=self.data,
                observ_mask=self.mask,
                input_step=self.input_step,
                pred_step=self.pred_step,
                block_size=self.block_size
            )
            # 复制一份用于验证，实际应划分
            self.val_dataset = TimeSeriesDataset(
                data=self.data,
                observ_mask=self.mask,
                input_step=self.input_step,
                pred_step=self.pred_step,
                block_size=self.block_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # DataLoader 会自动处理随机打乱
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
# test
if __name__ == '__main__':
    T, N, D = 100, 10, 5
    data = np.random.rand(T, N, D)
    observ_mask = np.random.randint(0, 2, (T, N, D))
    input_step, pred_step = 10, 5
    ds = TimeSeriesDataset(data, observ_mask, input_step, pred_step, block_size=None)
    print(ds[0])