# STCP/utils/synthetic_data.py
# (已修复: 移除了内部的归一化)

import numpy as np
from scipy.integrate import solve_ivp
import torch

# === 辅助函数 ===
def _generate_spatial_layout(n_nodes, layout_type='ring'):
    """
    为 n_nodes 个节点生成空间坐标 (locations)。
    """
    if layout_type == 'ring':
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        lat = np.cos(angles)
        lng = np.sin(angles)
        locations = np.stack([lat, lng], axis=0)
    elif layout_type == 'grid':
        grid_size = int(np.ceil(np.sqrt(n_nodes)))
        x = np.linspace(0, 1, grid_size)
        xx, yy = np.meshgrid(x, x)
        lat = xx.flatten()[:n_nodes]
        lng = yy.flatten()[:n_nodes]
        locations = np.stack([lat, lng], axis=0)
    elif layout_type == 'random':
        locations = np.random.rand(2, n_nodes)
    else:
        raise ValueError(f"Unknown spatial_layout: {layout_type}")
    return locations

# === lorenz 函数 ===
def lorenz96(t, x, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F
    return dxdt

# === generate_lorenz96 ===
def generate_lorenz96(n_nodes, num_steps_after_burn_in, num_steps_burn_in, F=10.0, delta_t=0.1, seed=0):
    """
    生成 Lorenz96 数据。
    """
    np.random.seed(seed)
    
    # 
    adj = np.eye(n_nodes, n_nodes)
    for i in range(n_nodes):
        adj[i, (i + 1) % n_nodes] = 1
        adj[i, (i - 1) % n_nodes] = 1
        adj[i, (i - 2) % n_nodes] = 1
    
    # 
    total_steps = num_steps_after_burn_in + num_steps_burn_in
    t_total_time = total_steps * delta_t
    
    t_span = [0, t_total_time]
    t_eval = np.arange(t_span[0], t_span[1], delta_t)
    
    # 
    x0 = np.random.rand(n_nodes) * 20 - 10
    
    sol = solve_ivp(lorenz96, t_span, x0, args=(F,), t_eval=t_eval, method='RK45')
    data = sol.y.T # 
    
    # 
    if data.shape[0] < total_steps:
        print(f"Warning: solve_ivp returned {data.shape[0]} steps, expected {total_steps}. Using all available steps.")
        num_steps_burn_in = min(num_steps_burn_in, int(data.shape[0] * 0.2)) 
    
    # 
    data = data[num_steps_burn_in:, :]
    
    # 
    locations = _generate_spatial_layout(n_nodes, 'ring')
    
    return data, adj, locations

# === var_stable ===
def var_stable(n_nodes, num_steps_after_burn_in, num_steps_burn_in, p, adj=None, noise_std=0.1, seed=0, spatial_layout='grid'):
    np.random.seed(seed)
    
    if adj is None:
        adj = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            adj[i, (i + 1) % n_nodes] = 1
            adj[i, (i - 1) % n_nodes] = 1
    
    A = np.random.randn(n_nodes, n_nodes * p) * 0.1
    # ... (VAR 内部生成逻辑无变化) ...
    for i in range(n_nodes):
        for j in range(n_nodes * p):
            if np.random.rand() > 0.5:
                A[i, j] = 0
    if p > 1:
        for i in range(n_nodes):
            for j in range(n_nodes * (p - 1), n_nodes * p):
                if adj[i, j % n_nodes] == 1:
                    A[i, j] = np.random.randn() * 0.5
    A_concat = np.zeros((n_nodes * p, n_nodes * p))
    A_concat[:n_nodes, :] = A
    for i in range(1, p):
        A_concat[i * n_nodes:(i + 1) * n_nodes, (i - 1) * n_nodes:i * n_nodes] = np.eye(n_nodes)
    eigvals = np.linalg.eigvals(A_concat)
    max_eigval = np.max(np.abs(eigvals))
    if max_eigval >= 1:
        A = A / (max_eigval + 0.1)
    
    # 
    total_steps = num_steps_after_burn_in + num_steps_burn_in
    
    data = np.zeros((total_steps + p, n_nodes)) # 
    noise = np.random.randn(total_steps + p, n_nodes) * noise_std
    
    for t in range(p, total_steps + p):
        for i in range(n_nodes):
            for j in range(n_nodes * p):
                data[t, i] += A[i, j] * data[t - (j // n_nodes) - 1, j % n_nodes]
            data[t, i] += noise[t, i]
            
    # 
    locations = _generate_spatial_layout(n_nodes, spatial_layout)
    
    # 
    data = data[p + num_steps_burn_in:, :]
            
    return data, adj, locations

# === 修改: generate_synthetic_data ===
def generate_synthetic_data(config, seed=0):
    """
    生成合成数据的主函数
    """
    n_nodes = config.n_nodes
    num_steps_after_burn_in = config.num_steps_after_burn_in
    num_steps_burn_in = config.num_steps_burn_in
    
    data_type = config.data_type
    spatial_layout = config.spatial_layout
    
    if data_type == 'lorenz':
        data, adj, locations = generate_lorenz96(
            n_nodes, num_steps_after_burn_in, num_steps_burn_in, 
            seed=seed
        )
    elif data_type == 'var':
        data, adj, locations = var_stable(
            n_nodes, num_steps_after_burn_in, num_steps_burn_in, 
            p=config.var_p, seed=seed, 
            spatial_layout=spatial_layout
        )
    else:
        raise ValueError(f"Unknown synthetic data type: {data_type}")

    data = data[..., np.newaxis] 
    
    return data, adj, locations