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
    def __init__(self, data_path: str, mask_path: str, config: dict):
        super().__init__()
        self.data_path = data_path
        self.mask_path = mask_path
        
        # 从配置中读取参数
        self.input_step = config.get("input_step")
        self.pred_step = config.get("pred_step")
        self.batch_size = config.get("batch_size")
        self.block_size = config.get("block_size", None)
        self.num_workers = config.get("num_workers", 4)
        
        self.data = None
        self.mask = None
        
        self.train_dataset = None
        self.val_dataset = None
        # 你可以根据需要添加 test_dataset

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

    def setup(self, stage: str = None):
        """
        加载和预处理数据，创建 Datasets
        """
        if self.data is None:
            # 假设数据是 .npy 文件，如果不是，请修改加载逻辑
            raw_data = np.load(self.data_path)
            self.mask = np.load(self.mask_path)
            
            # 确保数据是 (T, N, D)
            if raw_data.ndim == 2: # (T, N)
                raw_data = raw_data[..., None]
            if self.mask.ndim == 2: # (T, N)
                self.mask = self.mask[..., None]
            
            # 预处理（标准化）
            self.data = self._preprocess(raw_data)
        
        # 假设我们使用所有数据进行训练（原代码没有划分验证集）
        # 在实际应用中，你可以在这里划分 train/val/test
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