import torch
from torch import nn
from einops import rearrange

class ResidualMLP(nn.Module):
    """
    一个简单的残差 MLP 块，用于时间编码。
    它将替换 GRU。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        
        # 确保输入和输出维度匹配以进行残差连接
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
    时间编码器。
    它接收 (B, N, S, C) 格式的数据，将其展平并通过 MLP。
    """
    def __init__(self, input_step, in_channels, hidden_dim):
        super().__init__()
        
        # (S, C) -> (S * C)
        flattened_dim = input_step * in_channels
        
        self.encoder = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            ResidualMLP(hidden_dim * 2, hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # x: (B, N, S, C)
        # S = input_step, C = in_channels
        
        # 展平时间维度和通道维度
        # (B, N, S, C) -> (B, N, S*C)
        b, n, s, c = x.shape
        x_flat = rearrange(x, 'b n s c -> b n (s c)')
        
        # 通过 MLP 进行编码
        # (B, N, S*C) -> (B, N, D_hidden)
        h_temporal = self.encoder(x_flat)
        
        return h_temporal
    
#test
if __name__ == '__main__':
    B, N, S, C = 2, 3, 4, 5
    S_out = 2
    x = torch.randn(B, N, S, C)
    mlp = TemporalEncoder(S, C, S_out)
    y = mlp(x)
    print(y.shape)