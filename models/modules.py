import torch
import torch.nn as nn

# === 全新重构的代码 ===
# 这是对 PatchSTG/models/model.py 中使用的 timm 模块的原生 PyTorch 实现。

class NativeMLP(nn.Module):
    """
    原生 PyTorch 实现的 MLP (替换 timm.models.vision_transformer.Mlp)。
    逻辑来源: lmissher/patchstg/PatchSTG-feb68c369ac51fee2e730c2d27393bb9103c8a8c/models/model.py
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class NativeAttention(nn.Module):
    """
    原生 PyTorch 实现的 Attention (替换 timm.models.vision_transformer.Attention)。
    逻辑来源: lmissher/patchstg/PatchSTG-feb68c369ac51fee2e730c2d27393bb9103c8a8c/models/model.py
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, causal_mask=None):
        """
        Args:
            x (torch.Tensor): (B, N, D)
            causal_mask (torch.Tensor, optional): (B, H, N, N) or (B, 1, N, N)
        """
        B, N, D = x.shape
        # qkv: (B, N, 3 * D) -> (B, N, 3, H, D/H) -> (3, B, H, N, D/H)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, D/H)

        # attn: (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # *** STCP 核心: 应用因果掩码 ***
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x: (B, H, N, D/H) -> (B, N, H, D/H) -> (B, N, D)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class STCPBlock(nn.Module):
    """
    STCP 的核心 Transformer 块 (原生 PyTorch 实现)。
    结构改编自: lmissher/patchstg/PatchSTG-feb68c369ac51fee2e730c2d27393bb9103c8a8c/models/model.py 中的 WindowAttBlock。
    使用标准的 Pre-Norm 结构。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = NativeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = NativeMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop
        )

    def forward(self, x, causal_mask=None):
        # Pre-Norm
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.mlp(self.norm2(x))
        return x