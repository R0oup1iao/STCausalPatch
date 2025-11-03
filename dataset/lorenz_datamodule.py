import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from copy import deepcopy
from dataset.simu_data import simulate_lorenz_96, simulate_lorenz_96_patches, prepross_data
from utils.misc import reproduc


def generate_indices(input_step, pred_step, t_length, block_size=None):
    """
    从 cuts_plus.py 移过来的辅助函数，用于生成采样索引。
    """
    if block_size is None:
        block_size = t_length
        
    offsets_in_block = np.arange(input_step, block_size - pred_step + 1)
    if t_length % block_size != 0:
        print(f"Warning: t_length ({t_length}) % block_size ({block_size}) != 0")
        # 处理不能整除的情况
        num_blocks = t_length // block_size
        remaining = t_length % block_size
        
        random_t_list = []
        for block_start in range(0, num_blocks * block_size, block_size):
            random_t_list += (offsets_in_block + block_start).tolist()
        
        # 处理最后一个不完整的块
        if remaining > input_step + pred_step:
             offsets_in_remaining = np.arange(input_step, remaining - pred_step + 1)
             random_t_list += (offsets_in_remaining + num_blocks * block_size).tolist()
    
    else:
        random_t_list = []
        for block_start in range(0, t_length, block_size):
            random_t_list += (offsets_in_block + block_start).tolist()
            
    np.random.shuffle(random_t_list)
    return random_t_list


class TimeSeriesWindowDataset(Dataset):
    """
    一个标准的 PyTorch Dataset，用于替代 batch_generater。
    它接收 *可变* 的 data 和 mask 张量引用。
    """
    def __init__(self, data_tensor, mask_tensor, input_step, pred_step, block_size):
        super().__init__()
        self.data = data_tensor  # 这是一个张量引用
        self.mask = mask_tensor  # 这也是一个张量引用
        self.t_length, self.n_nodes, self.d_dim = data_tensor.shape
        
        self.input_step = input_step
        self.pred_step = pred_step
        self.block_size = block_size if block_size is not None else self.t_length

        # 生成采样的起始索引列表
        self.indices = generate_indices(input_step, pred_step, self.t_length, self.block_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # data_t 是 input window 的 *结束* 时间点
        data_t = self.indices[idx]
        
        # [B, N, T_in, D]
        x = self.data[data_t - self.input_step : data_t, :].permute(1, 0, 2)
        # [B, N, T_pred, D]
        y = self.data[data_t : data_t + self.pred_step, :].permute(1, 0, 2)
        
        # [B, N, T_in, D]
        mask_x = self.mask[data_t - self.input_step : data_t, :].permute(1, 0, 2)
        # [B, N, T_pred, D]
        mask_y = self.mask[data_t : data_t + self.pred_step, :].permute(1, 0, 2)

        return x, y, data_t, mask_x, mask_y

    def regenerate_indices(self):
        """ 在每个 epoch 开始时重新打乱索引 """
        self.indices = generate_indices(self.input_step, self.pred_step, self.t_length, self.block_size)


class Lorenz96DataModule(pl.LightningDataModule):
    """
    LightningDataModule 封装了所有数据准备和加载逻辑。
    """
    def __init__(self, data_config, train_config, reproduc_config):
        super().__init__()
        self.data_config = data_config
        self.train_config = train_config
        self.reproduc_config = reproduc_config
        
        self.batch_size = train_config.batch_size
        self.input_step = train_config.input_step
        self.pred_step = train_config.data_pred.pred_step
        self.block_size = train_config.get("block_size", None) # 使用 .get 获取可选参数

        # 这些张量将在 setup 中被初始化，并由 LightningModule *修改*
        self.train_data = None
        self.observ_mask = None
        self.original_data = None
        self.true_cm = None
        
        self.train_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # 应用复现性设置
            reproduc(**self.reproduc_config)
            
            # 1. 生成数据 (来自 cuts_plus_example.ipynb)
            if self.data_config.name == "lorenz_96":
                data, true_cm = simulate_lorenz_96(**self.data_config.param)
            elif self.data_config.name == "lorenz_96_patches":
                # 兼容补丁版数据集
                data, true_cm = simulate_lorenz_96_patches(**self.data_config.param)
            else:
                raise NotImplementedError(f"Data {self.data_config.name} not implemented")
            
            print(f"Data shape: {data.shape}")
            self.true_cm = true_cm
            self.original_data = torch.from_numpy(data).float() # 保存原始数据用于评估
            
            if self.train_config.n_nodes == "auto":
                # 如果是，用数据的实际节点数（data.shape[1]）替换它
                self.train_config.n_nodes = data.shape[1]
                print(f"Resolved 'n_nodes: auto' to {self.train_config.n_nodes}")

            # 2. 生成 Mask (来自 cuts_plus_example.ipynb)
            p_block = self.data_config.missing.params.p_block
            p_noise = self.data_config.missing.params.p_noise
            max_seq = self.data_config.missing.params.max_seq
            min_seq = self.data_config.missing.params.min_seq

            np.random.seed(self.data_config.seed)
            rand = np.random.random
            randint = np.random.randint

            init_mask = rand(data.shape) < p_block
            for col in range(init_mask.shape[1]):
                idxs = np.flatnonzero(init_mask[:, col])
                if not len(idxs):
                    continue
                fault_len = min_seq
                if max_seq > min_seq:
                    fault_len = fault_len + int(randint(max_seq - min_seq))
                idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
                idxs = np.unique(idxs_ext)
                idxs = np.clip(idxs, 0, init_mask.shape[0] - 1)
                init_mask[idxs, col] = True

            eval_mask = init_mask | (rand(init_mask.shape) < p_noise)
            mask_np = 1 - eval_mask
            
            # 3. 预处理和初始化
            data = data[:, :, None] # 增加维度
            mask_np = mask_np[:, :, None] # 增加维度
            
            # self.train_data 是将被模型插补和更新的数据
            # 初始时，它是带 mask 的数据 (原版使用 ZOH 插补，这里我们直接用 mask * data)
            # 注意：原版 prepross_data 在插补前执行，这里保持一致
            processed_data = prepross_data(data)
            self.train_data = torch.from_numpy(processed_data * mask_np).float()
            
            # self.observ_mask 是将被 fill_policy 和 supervision_policy 更新的 mask
            self.observ_mask = torch.from_numpy(mask_np).float()
            
            # 将原始数据也处理一下，用于后续 MSE 计算
            self.original_data = torch.from_numpy(prepross_data(data)).float()
            
            print(f"Initial observable data ratio: {self.observ_mask.mean().item():.4f}")

            # 4. 创建 Dataset
            self.train_dataset = TimeSeriesWindowDataset(
                self.train_data, 
                self.observ_mask,
                self.input_step,
                self.pred_step,
                self.block_size
            )

    def train_dataloader(self):
        # 在每个 epoch 开始时重新打乱索引
        self.train_dataset.regenerate_indices()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # DataLoader 会打乱 idx，从而间接打乱 data_t 的访问顺序
            num_workers=4,
            pin_memory=True
        )

    def update_data(self, data_pred_tensor, epoch):
        """
        由 LightningModule 在 on_train_epoch_end 调用。
        实现 'fill_policy' 逻辑。
        """
        fill_policy = self.train_config.fill_policy
        # 对齐设备，避免 cuda/cpu 混合运算
        if self.train_data is not None:
            data_pred_tensor = data_pred_tensor.to(self.train_data.device)
        
        if "every" in fill_policy:
            update_every = int(fill_policy.split("_")[-1])
            if (epoch + 1) % update_every == 0:
                print(f"Epoch {epoch+1}: Updating data with predictions (fill_policy: {fill_policy})")
                self.train_data = data_pred_tensor
                # "every" 策略也意味着 mask 变为全1
                self.observ_mask.fill_(1.0)
                
        elif "rate" in fill_policy:
            update_rate = float(fill_policy.split("_")[1])
            update_after = int(fill_policy.split("_")[3])
            if epoch + 1 > update_after:
                if epoch == update_after:
                    print(f"Epoch {epoch+1}: Starting data update (fill_policy: {fill_policy})")
                self.train_data = self.train_data * (1 - update_rate) + data_pred_tensor * update_rate
        else:
            # no data update
            pass

    def update_supervision(self, epoch):
        """
        由 LightningModule 在 on_train_epoch_start 调用。
        实现 'supervision_policy' 逻辑。
        """
        supervision_policy = self.train_config.supervision_policy
        
        if "masked_before" in supervision_policy:
            masked_before = int(supervision_policy.split("_")[2])
            if epoch == masked_before:
                print(f"Epoch {epoch}: Switching to full supervision (supervision_policy: {supervision_policy})")
                self.observ_mask.fill_(1.0)