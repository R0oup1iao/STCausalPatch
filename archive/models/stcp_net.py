import torch
import torch.nn as nn
from .modules import STCPBlock

# === 全新重构的代码 ===
# 逻辑主要改编自: lmissher/patchstg/PatchSTG-feb68c369ac51fee2e730c2d27393bb9103c8a8c/models/model.py

class STCPNet(nn.Module):
    """
    STCausalPatch (STCP) 主模型。
    (已重构，支持禁用 T/S 嵌入)
    """
    def __init__(self, config):
        super().__init__()
        self.n_nodes = config.model.n_nodes
        self.input_len = config.data.input_len   # P
        self.output_len = config.data.output_len # Q
        self.input_dim = config.data.input_dim

        self.embed_dim = config.model.embed_dim

        # --- 标志位 (来自 config) ---
        self.use_spatial_emb = config.model.use_spatial_emb
        self.use_temporal_emb = config.model.use_temporal_emb

        # 1. Spatio-Temporal Embedding (现在是可选的)

        # 1.1 Input Embedding
        # 
        input_fc_in_dim = self.input_dim
        if self.use_temporal_emb:
            input_fc_in_dim += 2 # 

        self.input_fc = nn.Conv2d(
            in_channels=input_fc_in_dim,
            out_channels=self.embed_dim, 
            kernel_size=(1, 1)
        )

        # 1.2 Spatial Embedding (可选)
        current_embed_dim = self.embed_dim
        if self.use_spatial_emb:
            self.node_emb_dim = config.model.node_emb_dim
            self.node_emb = nn.Parameter(torch.empty(self.n_nodes, self.node_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
            current_embed_dim += self.node_emb_dim

        # 1.3 Temporal Embedding (可选)
        if self.use_temporal_emb:
            self.tod_emb_dim = config.model.tod_emb_dim
            self.dow_emb_dim = config.model.dow_emb_dim
            self.tod_emb = nn.Parameter(torch.empty(config.model.tod_size, self.tod_emb_dim))
            nn.init.xavier_uniform_(self.tod_emb)
            self.dow_emb = nn.Parameter(torch.empty(config.model.dow_size, self.dow_emb_dim))
            nn.init.xavier_uniform_(self.dow_emb)
            current_embed_dim += self.tod_emb_dim + self.dow_emb_dim

        # 1.4 最终融合层
        # 
        self.embed_fusion_layer = nn.Linear(
            current_embed_dim,
            self.embed_dim
        )

        # 2. Causal Transformer Encoder (使用新的原生模块)
        self.encoder_layers = nn.ModuleList([
            STCPBlock(
                dim=self.embed_dim,
                num_heads=config.model.num_heads,
                mlp_ratio=config.model.mlp_ratio,
                qkv_bias=True,
                drop=config.model.dropout,
                attn_drop=config.model.attn_dropout
            ) for _ in range(config.model.num_layers)
        ])

        self.norm = nn.LayerNorm(self.embed_dim) # Encoder 后的最终 Norm

        # 3. Projection Decoder (逻辑来自 PatchSTG.regression_conv)
        self.regression_conv = nn.Conv2d(
            in_channels=self.input_len * self.embed_dim, 
            out_channels=self.output_len, 
            kernel_size=(1, 1), 
            bias=True
        )

    def _embed(self, x, te):
        # x: (B, T, N, D_in)
        # te: (B, T, N, 2)
        # 返回: (B, T, N, D_embed)

        B, T, N, _ = x.shape

        # 1.1 Input
        if self.use_temporal_emb:
            # 
            x_te = torch.cat([x, te[..., 0:1] / 288.0, te[..., 1:2] / 7.0], dim=-1).float()
        else:
            x_te = x # (B, T, N, D_in)

        x_te = x_te.permute(0, 3, 2, 1) # (B, D_in_f, N, T)
        input_data = self.input_fc(x_te) # (B, D_embed, N, T)
        input_data = input_data.permute(0, 3, 2, 1) # (B, T, N, D_embed)

        # 
        full_embed_list = [input_data]

        # 1.2 Time Embeddings (可选)
        if self.use_temporal_emb:
            te_tod = te[..., 0].long() # (B, T, N)
            te_dow = te[..., 1].long() # (B, T, N)

            tod_data = self.tod_emb[te_tod] # (B, T, N, D_tod)
            dow_data = self.dow_emb[te_dow] # (B, T, N, D_dow)
            full_embed_list.extend([tod_data, dow_data])

        # 1.3 Spatial Embedding (可选)
        if self.use_spatial_emb:
            node_data = self.node_emb.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1) # (B, T, N, D_node)
            full_embed_list.append(node_data)

        # 1.4 Fusion
        full_embed = torch.cat(full_embed_list, dim=-1)

        final_embed = self.embed_fusion_layer(full_embed) # (B, T, N, D_embed)
        return final_embed

    def forward(self, x, te, graph_sampled):
        """
        Args:
            x (torch.Tensor): (B, T_in, N, D_in) - T_in = P
            te (torch.Tensor): (B, T_in, N, 2)
            graph_sampled (torch.Tensor): (B, N, N) - 
        """
        B, T_in, N, _ = x.shape

        # 1. Embedding
        x_embed = self._embed(x, te) # (B, T_in, N, D_embed)

        # 2. Reshape for Transformer
        # (B, T, N, D) -> (B*T, N, D)
        x_tf = x_embed.reshape(B * T_in, N, self.embed_dim)

        # 3. 
        # graph_sampled: (B, N, N) -> (B, 1, N, N) -> (B*T_in, 1, N, N)
        causal_mask = graph_sampled.unsqueeze(1).repeat(T_in, 1, 1, 1)
        # 
        causal_mask = causal_mask.expand(-1, self.encoder_layers[0].attn.num_heads, -1, -1)

        # 4. Causal Transformer Encoder
        for layer in self.encoder_layers:
            x_tf = layer(x_tf, causal_mask=causal_mask)

        x_tf = self.norm(x_tf) # (B*T_in, N, D_embed)

        # 5. Projection Decoder 

        # 
        x_out = x_tf.reshape(B, T_in, N, self.embed_dim)

        # (B, T_in, N, D_embed) -> (B, T_in, D_embed, N)
        x_out = x_out.permute(0, 1, 3, 2)

        # (B, T_in * D_embed, N, 1)
        x_out = x_out.reshape(B, T_in * self.embed_dim, N, 1)

        # (B, Q, N, 1)
        pred_y = self.regression_conv(x_out)

        return pred_y