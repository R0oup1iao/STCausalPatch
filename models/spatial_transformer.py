# CUTS_Plus_Refactor/models/spatial.py
import torch
from torch import nn
import math

class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

    def generate_attention_mask(self, graph_sampled):
        """
        graph_sampled: (B, N, N)
        1 表示可以关注, 0 表示禁止
        Transformer 要求:
            True/-inf 表示 mask 禁止关注
            False/0   表示可关注
        """
        B, N, _ = graph_sampled.shape

        mask = (graph_sampled == 0).float() * -1e9  # (B, N, N)

        # Broadcast 给每个 multi-head attention
        mask = mask.repeat_interleave(self.n_heads, dim=0)  # (B*n_heads, N, N)
        return mask

    def forward(self, h_temporal, graph_sampled):
        # h_temporal: (B, N, D)
        # graph_sampled: (B, N, N)

        attn_mask = self.generate_attention_mask(graph_sampled)

        h_temporal = self.pos_encoder(h_temporal)  # (B, N, D)

        # mask --> (B*n_heads, N, N)
        h_spatial = self.transformer_encoder(h_temporal, mask=attn_mask)

        return h_spatial


class PositionalEncoding(nn.Module):
    # 标准的 Transformer 位置编码
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        # (B, N, D) -> (N, B, D) 
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        # (N, B, D) -> (B, N, D)
        return self.dropout(x)
    
# test
if __name__ == '__main__':
    B, N, D = 2, 4, 8

    model = SpatialTransformer(
        hidden_dim=D,
        n_heads=2,
        n_layers=2,
        dropout=0.1
    )

    # 输入特征 (B, N, D)
    h = torch.randn(B, N, D)

    # 因果图/邻接矩阵 (B, N, N)
    graph = torch.tensor([
        [
            [1,1,0,0],
            [1,1,1,0],
            [0,1,1,1],
            [0,0,1,1],
        ],
        [
            [1,0,0,1],
            [0,1,1,1],
            [0,1,1,0],
            [1,1,0,1],
        ]
    ], dtype=torch.float32)

    output = model(h, graph)

    print("input shape:", h.shape)
    print("output shape:", output.shape)
    print("output:", output)