import torch
from einops import rearrange
from torch import nn
from .spatial_transformer import SpatialTransformer


class MPNN(nn.Module):
    """保留MPNN作为向后兼容"""
    def __init__(self, c_in, c_out, concat_h=True):
        super(MPNN, self).__init__()
        self.concat_h = concat_h
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=1)
        
    def forward(self, x, h, graph):
        b, c, n = x.shape
        
        x_repeat = x[:, :, :, None].expand(-1, -1, -1, n) # [b, c, n, n]
        x_messages = torch.einsum('bcmn,bmn->bcmn', (x_repeat, graph))
        x_messages = rearrange(x_messages, 'b c m n -> b (c m) n')
        
        if self.concat_h:
            out = self.mlp(torch.cat([x_messages, h], dim=1))
        else:
            out = self.mlp(x_messages)
        return out


class TemporalMLPCell(nn.Module):
    """
    现代 MLP 单元，用于处理时间维度信息
    """
    
    def __init__(self, d_in, num_units, n_nodes, concat_h=False, mlp_depth=2, dropout=0.0):
        super(TemporalMLPCell, self).__init__()
        self.num_units = num_units
        self.concat_h = concat_h
        self.mlp_depth = mlp_depth
        
        # 输入投影：将图消息和隐藏状态融合
        mpnn_channel = d_in*n_nodes+num_units if concat_h else d_in*n_nodes
        self.input_proj = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        
        # LayerNorm 层
        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_units) for _ in range(mlp_depth)])
        
        # MLP 层：使用扩展-压缩结构
        self.mlp_up = nn.ModuleList()  # 扩展层
        self.mlp_down = nn.ModuleList()  # 压缩层
        self.dropouts = nn.ModuleList()
        
        for i in range(mlp_depth):
            self.mlp_up.append(nn.Conv1d(num_units, num_units * 2, kernel_size=1))
            self.mlp_down.append(nn.Conv1d(num_units * 2, num_units, kernel_size=1))
            self.dropouts.append(nn.Dropout(dropout))
        
        # 门控机制：学习如何融合当前输入和隐藏状态
        self.gate = nn.Sequential(
            nn.Conv1d(num_units * 2, num_units, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(num_units)
        
    def forward(self, x, h, adj):
        """
        :param x: (B, input_dim, num_nodes) - 当前时间步输入
        :param h: (B, num_units, num_nodes) - 隐藏状态
        :param adj: (B, num_nodes, num_nodes) 或 (num_nodes, num_nodes) - 邻接矩阵
        :return: (B, num_units, num_nodes) - 更新后的隐藏状态
        """
        # 1. 通过图消息传递融合输入和隐藏状态
        projected = self.input_proj(x, h, adj)  # (B, num_units, num_nodes)
        
        # 2. 通过多层 MLP 处理（带残差连接）
        residual = projected
        for i in range(self.mlp_depth):
            # LayerNorm (需要转置到 BNC 格式)
            projected_t = projected.transpose(1, 2)  # (B, N, C)
            normed_t = self.layer_norms[i](projected_t)
            normed = normed_t.transpose(1, 2)  # (B, C, N)
            
            # MLP: 扩展 -> GELU -> Dropout -> 压缩 -> Dropout
            expanded = self.mlp_up[i](normed)  # (B, 2*C, N)
            activated = torch.nn.functional.gelu(expanded)
            activated = self.dropouts[i](activated)
            compressed = self.mlp_down[i](activated)  # (B, C, N)
            compressed = self.dropouts[i](compressed)
            
            # 残差连接
            projected = compressed + residual
            residual = projected
        
        # 3. 门控融合：学习如何结合新的信息和历史信息
        combined = torch.cat([projected, h], dim=1)  # (B, 2*num_units, num_nodes)
        gate_weights = self.gate(combined)  # (B, num_units, num_nodes)
        
        # 4. 输出：门控融合 + LayerNorm
        output = gate_weights * projected + (1 - gate_weights) * h
        output_t = output.transpose(1, 2)  # (B, N, C)
        output = self.output_norm(output_t).transpose(1, 2)  # (B, C, N)
        
        return output


class GRUCell(nn.Module):
    """保留 GRUCell 作为向后兼容"""
    
    def __init__(self, d_in, num_units, n_nodes, concat_h=False, activation='tanh'):
        super(GRUCell, self).__init__()
        self.activation_fn = getattr(torch, activation)

        mpnn_channel = d_in*n_nodes+num_units if concat_h else d_in*n_nodes
        self.forget_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.update_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)
        self.c_gate = MPNN(c_in=mpnn_channel, c_out=num_units, concat_h=concat_h)

    def forward(self, x, h, adj):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        r = torch.sigmoid(self.forget_gate(x, h, adj))
        u = torch.sigmoid(self.update_gate(x, h, adj))
        c = self.c_gate(x, r * h, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        return u * h + (1. - u) * c


class LocalConv1D(nn.Module):
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
    """原始CUTS_Plus网络，使用MPNN处理空间关系"""
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
        x = rearrange(x, 'b n s c -> b c n s')
        bs, in_ch, n_nodes, steps = x.shape
        
        h = [h_.expand(bs, -1, -1) for h_ in self.h0.to(x.device)]
        
        pred = []
        for step in range(steps):
            x_now = x[..., step] # [batches, in_ch, nodes]
            
            """Update state"""
            h = self.update_state(x_now, h, fwd_graph)
            h_now = h[-1]
            
            """Prediction"""
            x_repr = self.act(self.conv_encoder1(h_now)) # [batches, hidden_ch, nodes]
            x_repr = self.act(self.conv_encoder2(torch.cat([x_repr, h_now], dim=1))) # [batches, hidden_ch, nodes]
            x_repr = torch.cat([x_repr, h_now], dim=1) # [batches, 2*hidden_ch, nodes]
            x_hat2 = self.decoder(x_repr) # [batches, in_ch, nodes]
            pred.append(x_hat2)
            
        
        pred = torch.stack(pred, dim=-1)
        pred = rearrange(pred, 'b c n s -> b n s c')
        return pred[:, :, -1:]


class CUTS_Plus_Transformer_Net(nn.Module):
    """
    使用Transformer替代MPNN的现代CUTS_Plus网络
    MLPCell处理时间维度后压缩成(B,N,C)向量，然后经过SpatialTransformer得到最终输出
    """
    
    def __init__(self, n_nodes,
                 in_ch=1,
                 hidden_ch=32,
                 n_layers=1,
                 shared_weights_decoder=False,
                 concat_h=False,
                 n_heads=4,
                 transformer_layers=2,
                 dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        
        # 时间维度处理：MLPCell
        self.cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(TemporalMLPCell(
                d_in=in_ch if i==0 else hidden_ch, 
                num_units=hidden_ch, 
                n_nodes=n_nodes,
                concat_h=concat_h,
                mlp_depth=2,
                dropout=dropout
            ))
        
        # 空间维度处理：SpatialTransformer
        self.spatial_transformer = SpatialTransformer(
            hidden_dim=hidden_ch,
            n_heads=n_heads,
            n_layers=transformer_layers,
            dropout=dropout
        )
        
        # 解码器
        if shared_weights_decoder:
            self.decoder = nn.Sequential(
                nn.Conv1d(in_channels=hidden_ch, out_channels=in_ch, kernel_size=1),
            )
        else:
            self.decoder = nn.Sequential(
                LocalConv1D(in_channels=hidden_ch, out_channels=in_ch, kernel_size=1, n_nodes=n_nodes),
            )
        
        self.act = nn.LeakyReLU()
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
        :param x: (B, N, T, C) - 输入数据
        :param mask: (B, N, T, C) - 掩码
        :param fwd_graph: (B, N, N) 或 (N, N) - 前向图
        :return: (B, N, 1, C) - 预测结果
        """
        x = rearrange(x, 'b n s c -> b c n s')
        bs, in_ch, n_nodes, steps = x.shape
        
        h = [h_.expand(bs, -1, -1) for h_ in self.h0.to(x.device)]
        
        pred = []
        for step in range(steps):
            x_now = x[..., step] # [batches, in_ch, nodes]
            
            """Update state: 时间维度处理"""
            h = self.update_state(x_now, h, fwd_graph)
            h_now = h[-1]  # (B, hidden_ch, N)
            
            """Spatial Transformer: 空间维度处理"""
            # 将(B, C, N)转换为(B, N, C)格式
            h_spatial = h_now.transpose(1, 2)  # (B, N, hidden_ch)
            
            # 通过SpatialTransformer处理空间关系
            h_spatial = self.spatial_transformer(h_spatial, fwd_graph)  # (B, N, hidden_ch)
            
            # 转换回(B, C, N)格式
            h_spatial = h_spatial.transpose(1, 2)  # (B, hidden_ch, N)
            
            """Prediction: 解码器"""
            x_hat2 = self.decoder(h_spatial)  # (B, in_ch, N)
            pred.append(x_hat2)
            
        
        pred = torch.stack(pred, dim=-1)
        pred = rearrange(pred, 'b c n s -> b n s c')
        return pred[:, :, -1:]
