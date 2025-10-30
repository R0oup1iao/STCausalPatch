# models/temporal_mlp.py
import torch
from torch import nn
from einops import rearrange

class ResidualBlock(nn.Module):
    """ 一个简单的残差 MLP 块 """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.act(self.mlp1(x))
        h = self.dropout(self.mlp2(h))
        return self.act(h + self.shortcut(x))

class TemporalEncoder(nn.Module):
    """
    时间编码器 (替换 GRU)。
    它接收 (B, N, S, C) 数据, 将其展平为 (B, N, S*C),
    然后通过 MLP 将其编码为 (B, N, D_hidden)
    """
    def __init__(self, input_step, in_channels, hidden_dim, mlp_hidden_dim, num_blocks=1):
        super().__init__()
        
        flattened_dim = input_step * in_channels
        
        layers = [
            nn.Linear(flattened_dim, mlp_hidden_dim),
            nn.LeakyReLU()
        ]
        
        for _ in range(num_blocks):
            layers.append(ResidualBlock(mlp_hidden_dim, mlp_hidden_dim, mlp_hidden_dim))
            
        layers.append(nn.Linear(mlp_hidden_dim, hidden_dim))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, S, C)
        # S = input_step, C = in_channels
        
        # 展平时间维度和通道维度
        # (B, N, S, C) -> (B, N, S*C)
        x_flat = rearrange(x, 'b n s c -> b n (s c)')
        
        # 通过 MLP 进行编码
        # (B, N, S*C) -> (B, N, D_hidden)
        h_temporal = self.encoder(x_flat)
        
        return h_temporal