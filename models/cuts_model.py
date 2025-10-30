import torch
from torch import nn
from .temporal_mlp import ResidualMLP
from .spatial_transformer import SpatialTransformer

class CUTSModel(nn.Module):
    """
    组合了 TemporalMLP 和 SpatialTransformer 的新预测模型
    """
    def __init__(self, cfg_model, cfg_data):
        super().__init__()
        self.n_nodes = cfg_data.get("n_nodes")
        self.in_channels = cfg_data.get("data_dim")
        self.input_step = cfg_data.get("input_step")
        self.pred_step = cfg_data.get("pred_step")

        self.temporal_hidden = cfg_model.get("temporal_hidden_dim")
        self.embed_dim = cfg_model.get("spatial_embed_dim")

        # 1. 时间编码器 (替换 GRU)
        # (B, N, C, S_in) -> (B, N, C, H_t)
        self.temporal_encoder = ResidualMLP(
            seq_len_in=self.input_step,
            seq_len_out=self.temporal_hidden,
            channel_in=self.in_channels,
            channel_hidden=cfg_model.get("temporal_mlp_hidden")
        )

        # 2. 特征投影
        # (B, N, C, H_t) -> (B, N, E)
        self.input_projection = nn.Linear(
            self.in_channels * self.temporal_hidden,
            self.embed_dim
        )
        
        # 3. 空间编码器 (替换 MPNN)
        self.spatial_encoder = SpatialTransformer(
            embed_dim=self.embed_dim,
            n_heads=cfg_model.get("spatial_n_heads"),
            dim_feedforward=cfg_model.get("spatial_ffn_dim"),
            n_layers=cfg_model.get("spatial_n_layers")
        )

        # 4. 预测头
        # (B, N, E) -> (B, N, P * C)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.embed_dim, cfg_model.get("prediction_head_hidden")),
            nn.ReLU(),
            nn.Linear(cfg_model.get("prediction_head_hidden"), self.pred_step * self.in_channels)
        )

    def forward(self, x, x_mask, adj_matrix):
        """
        :param x: (B, N, S, C)
        :param x_mask: (B, N, S, C)
        :param adj_matrix: (B, N, N)
        :return: y_pred (B, N, P, C)
        """
        
        # 应用掩码 (如果需要，原代码是在 loss 中应用)
        # x = x * x_mask
        
        # (B, N, S, C) -> (B, N, C, S)
        x = x.permute(0, 1, 3, 2)
        
        # 1. Temporal Encoding
        # (B, N, C, S) -> (B, N, C, H_t)
        x_temporal = self.temporal_encoder(x)
        
        # 2. Feature Projection
        # (B, N, C, H_t) -> (B, N, C * H_t)
        b, n, c, h_t = x_temporal.shape
        x_flat = x_temporal.reshape(b, n, -1)
        
        # (B, N, C * H_t) -> (B, N, E)
        x_embed = self.input_projection(x_flat)
        
        # 3. Spatial Encoding
        # (B, N, E) -> (B, N, E)
        x_spatial = self.spatial_encoder(x_embed, adj_matrix)
        
        # 4. Prediction Head
        # (B, N, E) -> (B, N, P * C)
        y_pred_flat = self.prediction_head(x_spatial)
        
        # (B, N, P * C) -> (B, N, P, C)
        y_pred = y_pred_flat.view(b, n, self.pred_step, self.in_channels)
        
        return y_pred