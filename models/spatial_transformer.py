import torch
import torch.nn as nn
import math


class SpatialTransformer(nn.Module):
    """
    现代Transformer模块，用于替换MPNN处理空间关系
    通过batch attention mask完美替代原有的graph结构
    """
    
    def __init__(self, hidden_dim, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            norm_first=True,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

    def generate_attention_mask(self, graph):
        """
        将graph转换为Transformer的attention mask
        graph: (B, N, N) 或 (N, N)
        1表示可以关注，0表示禁止
        Transformer要求：
            True/-inf 表示mask禁止关注
            False/0   表示可关注
        """
        if len(graph.shape) == 2:
            # (N, N) -> (1, N, N)
            graph = graph.unsqueeze(0)
        
        B, N, _ = graph.shape
        
        # 将graph转换为attention mask
        # graph中1表示有连接，0表示无连接
        # 我们需要将无连接的位置mask掉
        mask = (graph == 0).float() * -1e9  # (B, N, N)

        # Broadcast给每个multi-head attention
        mask = mask.repeat_interleave(self.n_heads, dim=0)  # (B*n_heads, N, N)
        return mask

    def forward(self, x, graph):
        """
        :param x: (B, N, D) - 输入特征，已经压缩成(B,N,C)格式
        :param graph: (B, N, N) 或 (N, N) - 邻接矩阵
        :return: (B, N, D) - 更新后的表示
        """
        # 生成attention mask
        attn_mask = self.generate_attention_mask(graph)
        
        # 添加位置编码
        x = self.pos_encoder(x)  # (B, N, D)

        # 通过Transformer编码器
        output = self.transformer_encoder(x, mask=attn_mask)

        return output


class PositionalEncoding(nn.Module):
    """
    标准的Transformer位置编码
    """
    
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


# 测试代码
if __name__ == '__main__':
    B, N, D = 2, 4, 32

    model = SpatialTransformer(
        hidden_dim=D,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    )

    # 输入特征 (B, N, D)
    x = torch.randn(B, N, D)

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

    output = model(x, graph)

    print("input shape:", x.shape)
    print("graph shape:", graph.shape)
    print("output shape:", output.shape)
    print("output:", output)
