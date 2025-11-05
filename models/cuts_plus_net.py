import torch
from einops import rearrange
from torch import nn


class SpatialTransformer(nn.Module):
    """
    (V4 修复版)
    
    1. 强制 mask 对角线为 -inf, 禁止自注意力. 
       Transformer 变为纯粹的 "他人信息聚合器".
    2. (在 CUTS_Plus_Transformer_Net 中) 将聚合
       的 "他人信息" (h_spatial) 与 "自身信息"
       (h_temporal) 相加后送入 Decoder。
    """
    
    def __init__(self, hidden_dim, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.MultiheadAttention(
                    embed_dim=hidden_dim, 
                    num_heads=n_heads, 
                    dropout=dropout, 
                    batch_first=True
                )
            ]))
        
        self.final_norm = nn.LayerNorm(hidden_dim)

    def generate_attention_mask(self, graph):
        """
        将graph转换为Transformer的attention mask
        graph: (B, N, N) 或 (N, N)
        """
        if len(graph.shape) == 2:
            graph = graph.unsqueeze(0)
        
        B, N, _ = graph.shape
        
        # 1. 基础 mask: 1 表示 attend, 0 表示 mask
        base_mask = (graph == 0).float() * -1e9  # (B, N, N)

        # 2. --- 关键修改 (V4) ---
        #    强制移除自注意力, 无论 graph[i, i] 是什么.
        #    这迫使 h_spatial 成为纯粹的 "他人信息".
        base_mask.diagonal(dim1=-2, dim2=-1).fill_(-torch.inf)
        # ----------------------

        # Broadcast给每个multi-head attention
        mask = base_mask.repeat_interleave(self.n_heads, dim=0)  # (B*n_heads, N, N)
        return mask

    def forward(self, x, graph):
        """
        :param x: (B, N, D) - 输入特征 (h_temporal)
        :param graph: (B, N, N) - 邻接矩阵
        :return: (B, N, D) - "他人" 信息的聚合
        """
        attn_mask = self.generate_attention_mask(graph)
        
        h = x
        for norm, attn in self.layers:
            h_norm = norm(h)
            
            # (移除了残差连接)
            h, _ = attn(
                h_norm, h_norm, h_norm, 
                attn_mask=attn_mask,
                need_weights=False
            )
            
        return self.final_norm(h)


class PositionalEncoding(nn.Module):
    """ (保留, 以防将来使用) """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class MPNN(nn.Module):
    """(保留) MPNN, 为 CUTS_Plus_Net 服务"""
    def __init__(self, c_in, c_out, concat_h=True):
        super(MPNN, self).__init__()
        self.concat_h = concat_h
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=1)
        
    def forward(self, x, h, graph):
        b, c, n = x.shape
        x_repeat = x[:, :, :, None].expand(-1, -1, -1, n)
        x_messages = torch.einsum('bcmn,bmn->bcmn', (x_repeat, graph))
        x_messages = rearrange(x_messages, 'b c m n -> b (c m) n')
        
        if self.concat_h:
            out = self.mlp(torch.cat([x_messages, h], dim=1))
        else:
            out = self.mlp(x_messages)
        return out


class TemporalMLPCell(nn.Module):
    """ (保留) 现代 MLP 单元, 为 CUTS_Plus_Net 服务 """
    def __init__(self, d_in, num_units, n_nodes, concat_h=False, mlp_depth=2, dropout=0.0):
        super(TemporalMLPCell, self).__init__()
        self.num_units = num_units
        self.concat_h = concat_h
        self.mlp_depth = mlp_depth
        
        mpnn_channel = d_in*n_nodes+num_units if concat_h else d_in*n_nodes
        self.input_proj = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_units) for _ in range(mlp_depth)])
        self.mlp_up = nn.ModuleList()
        self.mlp_down = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(mlp_depth):
            self.mlp_up.append(nn.Conv1d(num_units, num_units * 2, kernel_size=1))
            self.mlp_down.append(nn.Conv1d(num_units * 2, num_units, kernel_size=1))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.gate = nn.Sequential(
            nn.Conv1d(num_units * 2, num_units, kernel_size=1),
            nn.Sigmoid()
        )
        self.output_norm = nn.LayerNorm(num_units)
        
    def forward(self, x, h, adj):
        projected = self.input_proj(x, h, adj)
        residual = projected
        for i in range(self.mlp_depth):
            projected_t = projected.transpose(1, 2)
            normed_t = self.layer_norms[i](projected_t)
            normed = normed_t.transpose(1, 2)
            expanded = self.mlp_up[i](normed)
            activated = torch.nn.functional.gelu(expanded)
            activated = self.dropouts[i](activated)
            compressed = self.mlp_down[i](activated)
            compressed = self.dropouts[i](compressed)
            projected = compressed + residual
            residual = projected
        
        combined = torch.cat([projected, h], dim=1)
        gate_weights = self.gate(combined)
        
        output = gate_weights * projected + (1 - gate_weights) * h
        output_t = output.transpose(1, 2)
        output = self.output_norm(output_t).transpose(1, 2)
        return output


class GRUCell(nn.Module):
    """(保留) 原始 GRUCell"""
    def __init__(self, d_in, num_units, n_nodes, concat_h=False, activation='tanh'):
        super(GRUCell, self).__init__()
        self.activation_fn = getattr(torch, activation)
        mpnn_channel = d_in*n_nodes+num_units if concat_h else d_in*n_nodes
        self.forget_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.update_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.c_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)

    def forward(self, x, h, adj):
        r = torch.sigmoid(self.forget_gate(x, h, adj))
        u = torch.sigmoid(self.update_gate(x, h, adj))
        c = self.c_gate(x, r * h, adj)
        c = self.activation_fn(c)
        return u * h + (1. - u) * c


class LocalConv1D(nn.Module):
    """ (保留) 局部卷积, 两个模型都在用 """
    def __init__(self, in_channels, out_channels, kernel_size, n_nodes):
        super(LocalConv1D, self).__init__()
        self.out_channel = out_channels
        self.conv_list = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size) for _ in range(n_nodes)
        ])
    
    def forward(self, x): # x: [batch, features, nodes]
        b, h, n = x.shape
        out = torch.zeros((b, self.out_channel, n)).to(x.device)
        for i in range(n):
            x_local_in = x[..., i].unsqueeze(-1)
            x_local_out = self.conv_list[i](x_local_in)
            out[..., i] = x_local_out.squeeze(-1)
        return out


class CUTS_Plus_Net(nn.Module):
    """
    (保留) 原始CUTS_Plus网络 (RNN 架构)
    - 接口已更新, forward 方法正确使用 mask
    - (V6) 注意: 此模型仍为 "有状态" (Stateful)
    """
    def __init__(self, n_nodes,
                 in_ch=1,
                 hidden_ch=32,
                 n_layers=1,
                 shared_weights_decoder=False,
                 concat_h=False,):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.n_layers = n_layers
        
        self.conv_encoder1 = nn.Conv1d(in_channels=hidden_ch, out_channels=hidden_ch, kernel_size=1)
        self.conv_encoder2 = nn.Conv1d(in_channels=2*hidden_ch, out_channels=hidden_ch, kernel_size=1)
        if shared_weights_decoder:
            self.decoder = nn.Sequential(
                nn.Conv1d(in_channels=2*hidden_ch, out_channels=in_ch, kernel_size=1),
            )
        else:
            self.decoder = nn.Sequential(
                LocalConv1D(in_channels=2*hidden_ch, out_channels=in_ch, kernel_size=1, n_nodes=n_nodes),
            )
        self.act = nn.LeakyReLU()
        
        self.cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(TemporalMLPCell(d_in=in_ch if i==0 else hidden_ch, 
                                            num_units=hidden_ch, 
                                            n_nodes=n_nodes,
                                            concat_h=concat_h,
                                            mlp_depth=2,
                                            dropout=0.0))
            
        self.h0 = self.init_state(n_nodes)
    
    def init_state(self, n_nodes):
        h = []
        for layer in range(self.n_layers):
            h.append(nn.parameter.Parameter(torch.zeros([self.hidden_ch, n_nodes])))
        return nn.ParameterList(h)
    
    def update_state(self, x, h, graph):
        rnn_in = x
        for layer in range(self.n_layers):
            rnn_in = h[layer] = self.cells[layer](rnn_in, h[layer], graph)
        return h
    
    def forward(self, x, mask, fwd_graph):
        """
        :param x: (B, N, T_hist, C)
        :param mask: (B, N, T_hist, C)
        :param fwd_graph: (B, N, N)
        :return: (B, N, 1, C)
        """
        # (V6) 即使 T=10, RNN 也只关心最后一步
        # (B, N, T, C) -> (B, C, N, T)
        x = rearrange(x, 'b n s c -> b c n s')
        mask = rearrange(mask, 'b n s c -> b c n s')
        
        bs, in_ch, n_nodes, steps = x.shape
        
        h = [h_.expand(bs, -1, -1) for h_ in self.h0.to(x.device)]
        
        # (V6) RNN 循环, 但只取最后一步
        # 注意: 这是一种低效的 RNN 实现, 但忠于原版
        for step in range(steps):
            x_now = x[..., step]
            mask_now = mask[..., step]
            h = self.update_state(x_now * mask_now, h, fwd_graph)
        
        # (V6) 只使用 RNN 最终的 h
        h_now = h[-1]
        x_repr = self.act(self.conv_encoder1(h_now))
        x_repr = self.act(self.conv_encoder2(torch.cat([x_repr, h_now], dim=1)))
        x_repr = torch.cat([x_repr, h_now], dim=1)
        x_hat2 = self.decoder(x_repr) # (B, C, N)
        
        # (B, C, N) -> (B, N, 1, C)
        pred = rearrange(x_hat2, 'b c n -> b n 1 c')
        return pred


# -----------------------------------------------------------------
# --- (V6) "无状态" (Stateless) Transformer (基于 V4) ---
# --- (V6) 匹配 T=10 的数据设置 ---
# -----------------------------------------------------------------
class CUTS_Plus_Transformer_Net(nn.Module):
    """
    (V6) "无状态" (Stateless) 架构
    - 匹配 T=10 的数据
    - 使用 TemporalEncoder 压缩 T
    - 使用 V4 SpatialTransformer 分离 "自身" vs "他人"
    """
    def __init__(self, n_nodes,
                 in_ch=1,
                 hidden_ch=32,
                 shared_weights_decoder=False,
                 n_heads=4,
                 transformer_layers=2,
                 dropout=0.0,
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.n_nodes = n_nodes

        # 1. 时间编码器 (Temporal Encoder)
        #    (V6) 现在 T=10, Conv1d(k=3) 可以工作了
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=hidden_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 2. 空间维度处理：SpatialTransformer (V4 "无作弊"版)
        self.spatial_transformer = SpatialTransformer(
            hidden_dim=hidden_ch,
            n_heads=n_heads,
            n_layers=transformer_layers,
            dropout=dropout
        )
        
        # 3. 解码器 (Decoder)
        if shared_weights_decoder:
            self.decoder = nn.Sequential(
                nn.Conv1d(in_channels=hidden_ch, out_channels=in_ch, kernel_size=1),
            )
        else:
            self.decoder = nn.Sequential(
                LocalConv1D(in_channels=hidden_ch, out_channels=in_ch, kernel_size=1, n_nodes=n_nodes),
            )
        
        self.norm_temporal = nn.LayerNorm(hidden_ch)
    
    def forward(self, x, mask, fwd_graph):
        """
        :param x: (B, N, T_hist=10, C)
        :param mask: (B, N, T_hist=10, C)
        :param fwd_graph: (B, N, N)
        :return: (B, N, 1, C)
        """
        B, N, T_hist, C_in = x.shape
        
        # 1. 应用 Mask
        x = x * mask

        # 2. 时间编码 (自身信息)
        # (B, N, T, C) -> (B, N, C, T) -> (B*N, C, T)
        x_t = rearrange(x, 'b n t c -> (b n) c t')
        
        # (B*N, C, T) -> (B*N, D, 1) -> (B, N, D)
        h_temporal = self.temporal_encoder(x_t)
        h_temporal = rearrange(h_temporal, '(b n) d 1 -> b n d', b=B, n=N)
        
        # 3. 空间 Transformer (他人信息)
        #    (V4: 内部已禁止自注意力)
        h_spatial = self.spatial_transformer(h_temporal, fwd_graph)
        
        # 4. 组合 "自身" 与 "他人" 信息
        h_combined = self.norm_temporal(h_temporal) + h_spatial
        
        # 5. 解码器
        h_combined_t = rearrange(h_combined, 'b n d -> b d n')
        y_pred = self.decoder(h_combined_t)
        
        # 6. 格式化输出 (B, C, N) -> (B, N, 1, C)
        y_pred = rearrange(y_pred, 'b c n -> b n 1 c')
        
        return y_pred