import torch
import numpy as np
import pandas as pd

def read_meta(path):
    """
    读取 meta.csv 文件。
    """
    meta = pd.read_csv(path)
    lat = meta['Lat'].values
    lng = meta['Lng'].values
    locations = np.stack([lat,lng], 0)
    return locations

def kdTree(locations, times, axis):
    """
    KDTree 递归划分函数。
    """
    # locations: [2,N] contains lng and lat
    # times: depth of kdtree
    # axis: select lng or lat as hyperplane to split points
    sorted_idx = np.argsort(locations[axis])
    part1, part2 = np.sort(sorted_idx[:locations.shape[1]//2]), np.sort(sorted_idx[locations.shape[1]//2:])
    parts = []
    if times == 1:
        return [part1, part2]
    else:
        left_parts = kdTree(locations[:,part1], times-1, axis^1)
        right_parts = kdTree(locations[:,part2], times-1, axis^1)
        for part in left_parts:
            parts.append(part1[part])
        for part in right_parts:
            parts.append(part2[part])
    return parts

# === 新增的重构代码 ===

class HierarchicalKDTreePartitioner:
    """
    用于管理分层 KDTree 划分的重构类。
    """
    def __init__(self, locations, config):
        """
        初始化划分器。
        Args:
            locations (np.array): [2, N] 的经纬度数组 (来自 read_meta)。
            config (OmegaConf): 包含 spatial.* 配置的对象。
        """
        self.locations = locations
        self.n_nodes = locations.shape[1]
        self.initial_depth = config.initial_depth
        self.final_depth = config.final_depth

        # 确保 final_depth 足够深，可以划分到单个节点
        max_needed_depth = int(np.ceil(np.log2(self.n_nodes)))
        if self.final_depth < max_needed_depth:
            print(f"Warning: final_depth ({self.final_depth}) is less than needed ({max_needed_depth}) to reach single nodes. Adjusting...")
            self.final_depth = max_needed_depth

        self.schedule_type = config.schedule_type

    def get_current_depth(self, epoch, max_epoch):
        """
        根据 epoch 计算当前的 KDTree 递归深度。
        """
        if self.schedule_type == 'linear':
            # 线性增长
            schedule_progress = min(1.0, epoch / max_epoch)
            depth_range = self.final_depth - self.initial_depth
            current_depth_float = self.initial_depth + schedule_progress * depth_range
            current_depth = int(np.round(current_depth_float))
        else:
            # 默认为线性
            schedule_progress = min(1.0, epoch / max_epoch)
            depth_range = self.final_depth - self.initial_depth
            current_depth = int(np.round(self.initial_depth + schedule_progress * depth_range))

        return current_depth

    def get_partitions(self, current_depth):
        """
        使用 kdTree 获取当前深度的分区。
        """
        if current_depth == 0:
            # 特殊情况：深度为0，所有节点在一个组
            return [np.arange(self.n_nodes)]

        # 
        parts_idx = kdTree(self.locations, current_depth, axis=0)
        return parts_idx

    def partitions_to_group_matrix(self, partitions, device):
        """
        将分区列表转换为 (N_nodes, N_groups) 的二元矩阵 G。
        """
        num_groups = len(partitions)
        G = torch.zeros(self.n_nodes, num_groups).to(device)

        for i, part_indices in enumerate(partitions):
            G[part_indices, i] = 1

        return G