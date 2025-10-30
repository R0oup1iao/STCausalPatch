# models/cuts_model.py
import torch
from torch import nn
from .temporal_mlp import TemporalEncoder # 导入我们刚修复的 TemporalEncoder
from .spatial_transformer import SpatialTransformer # 导入 spatial_transformer

class CUTSModel(nn.Module):
    """
    组合了 TemporalEncoder 和 SpatialTransformer 的新预测模型
    (已修复数据流)
    """
    def __init__(self, cfg_model, cfg_data):
        super().__init__()
        self.n_nodes = cfg_data.get("n_nodes")
        self.in_channels = cfg_data.get("data_dim")
        self.input_step = cfg_data.get("input_step")
        self.pred_step = cfg_data.get("pred_step")

        self.embed_dim = cfg_model.get("embed_dim") # 统一的嵌入维度

        # 1. 时间编码器 (替换 GRU)
        # (B, N, S, C) -> (B, N, E)
        self.temporal_encoder = TemporalEncoder(
            input_step=self.input_step,
            in_channels=self.in_channels,
            hidden_dim=self.embed_dim, # 输出统一的嵌入维度
            mlp_hidden_dim=cfg_model.get("temporal_mlp_hidden")
        )
        
        # 2. 空间编码器 (替换 MPNN)
        # (B, N, E) -> (B, N, E)
        self.spatial_encoder = SpatialTransformer(
            hidden_dim=self.embed_dim,
            n_heads=cfg_model.get("spatial_n_heads"),
            n_layers=cfg_model.get("spatial_n_layers")
        )

        # 3. 预测头
        # (B, N, E) -> (B, N, P * C)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.embed_dim, cfg_model.get("prediction_head_hidden")),
            nn.ReLU(),
            nn.Linear(cfg_model.get("prediction_head_hidden"), self.pred_step * self.in_channels)
        )

    def forward(self, x, x_mask, adj_matrix):
        """
        :param x: (B, N, S, C)  <-- 来自 DataLoader
        :param x_mask: (B, N, S, C)
        :param adj_matrix: (B, N, N)
        :return: y_pred (B, N, P, C)
        """
        b, n, s, c = x.shape
        
        # 1. Temporal Encoding
        # (B, N, S, C) -> (B, N, E)
        # 我们不再需要 .permute()
        x_temporal_embed = self.temporal_encoder(x)
        
        # 2. Spatial Encoding
        # (B, N, E) -> (B, N, E)
        x_spatial_embed = self.spatial_encoder(x_temporal_embed, adj_matrix)
        
        # 3. Prediction Head
        # (B, N, E) -> (B, N, P * C)
        y_pred_flat = self.prediction_head(x_spatial_embed)
        
        # (B, N, P * C) -> (B, N, P, C)
        y_pred = y_pred_flat.view(b, n, self.pred_step, self.in_channels)
        
        return y_pred
    

def test_cuts_model_forward():
    """
    测试 CUTSModel 的前向传播逻辑是否正常工作
    """

    # ===== 配置模拟 (仿照 config.yaml) ===== #
    cfg_model = {
        "embed_dim": 64,                  # TemporalEncoder 和 SpatialTransformer 输出维度必须一致
        "temporal_mlp_hidden": 128,      # Temporal MLP 隐层维度
        "spatial_n_heads": 4,            # Transformer 多头注意力
        "spatial_ffn_dim": 128,          # Transformer FFN 隐层
        "spatial_n_layers": 2,           # Transformer 编码层数
        "prediction_head_hidden": 64
    }

    cfg_data = {
        "n_nodes": 12,                   # 节点数，例如有 12 个传感器
        "data_dim": 3,                   # 每个时间步的特征维度，例如速度/车流量/密度
        "input_step": 8,                 # 输入历史序列长度
        "pred_step": 5                   # 要预测的未来步数
    }

    # ===== 构造测试输入数据 ===== #
    B = 2                                # batch size
    N = cfg_data["n_nodes"]
    S = cfg_data["input_step"]
    C = cfg_data["data_dim"]
    P = cfg_data["pred_step"]

    x = torch.randn(B, N, S, C)          # 模拟输入 (B, N, S, C)
    x_mask = torch.ones_like(x)          # mask 一般是 0/1，这里先不做逻辑处理
    adj = torch.randn(B, N, N)           # 随便给个邻接矩阵

    # ===== 构造模型 ===== #
    model = CUTSModel(cfg_model, cfg_data)

    # ===== 执行前向传播 ===== #
    y_pred = model(x, x_mask, adj)

    print("input shape:", x.shape)
    print("output shape:", y_pred.shape)

    # ===== 检查输出是否符合预期形状 ===== #
    assert y_pred.shape == (B, N, P, C), \
        f"模型输出形状错误: got {y_pred.shape}, expected {(B, N, P, C)}"

    print("\n✅ 测试通过！")

if __name__ == "__main__":
    test_cuts_model_forward()
