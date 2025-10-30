# STCP/utils/data_loader.py
# (已修复 Bug: 归一化逻辑)
# (已修复 Log: 澄清 4D 形状)

import torch
import numpy as np
import os
from .spatial import read_meta
from .synthetic_data import generate_synthetic_data
from .logging import log_string

# === seq2instance (无变化) ===
def seq2instance(data, P, Q):
    """
    将时序数据转换为 (X, Y) 样本对。
    """
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    y = np.zeros(shape = (num_sample, Q, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

# === _load_real_world_dataset (无变化) ===
def _load_real_world_dataset(config_data, config_model, log_f):
    """
    加载真实世界时空数据集。
    """
    Traffic = np.load(config_data.traffic_file)['data'][..., :config_data.input_dim]
    locations = read_meta(config_data.meta_file)
    num_step, num_nodes = Traffic.shape[0], Traffic.shape[1]
    
    tod = config_model.tod_size
    dow = config_model.dow_size
    TE = np.zeros([num_step, 2])
    TE[:,0] = np.array([i % tod for i in range(num_step)])
    TE[:,1] = np.array([(i // tod) % dow for i in range(num_step)])
    TE_tile = np.repeat(np.expand_dims(TE, 1), num_nodes, 1)

    log_string(log_f, f'Shape of data: {Traffic.shape}')
    log_string(log_f, f'Shape of locations: {locations.shape}')

    train_steps = round(config_data.train_ratio * num_step)
    val_steps = round(config_data.val_ratio * num_step)
    test_steps = num_step - train_steps - val_steps

    trainData, trainTE = Traffic[: train_steps], TE_tile[: train_steps]
    valData, valTE = Traffic[train_steps : train_steps + val_steps], TE_tile[train_steps : train_steps + val_steps]
    testData, testTE = Traffic[-test_steps :], TE_tile[-test_steps :]

    P, Q = config_data.input_len, config_data.output_len
    trainX, trainY = seq2instance(trainData, P, Q)
    valX, valY = seq2instance(valData, P, Q)
    testX, testY = seq2instance(testData, P, Q)
    
    trainXTE, trainYTE = seq2instance(trainTE, P, Q)
    valXTE, valYTE = seq2instance(valTE, P, Q)
    testXTE, testYTE = seq2instance(testTE, P, Q)

    # 
    mean, std = np.mean(trainX), np.std(trainX)

    #!#!#! 修复: 澄清日志输出
    log_string(log_f, "Data shapes (Samples, Time, Nodes, Features):")
    log_string(log_f, f'Shape of Train X (P={P}): {trainX.shape}')
    log_string(log_f, f'Shape of Train Y (Q={Q}): {trainY.shape}')
    log_string(log_f, f'Shape of Validation Y (Q={Q}): {valY.shape}')
    log_string(log_f, f'Shape of Test Y (Q={Q}): {testY.shape}')
    log_string(log_f, f'Mean (Train X): {mean} & Std (Train X): {std}')

    dataset_pack = {
        'data': {
            'trainX': trainX, 'trainY': trainY, 'trainXTE': trainXTE,
            'valX': valX, 'valY': valY, 'valXTE': valXTE,
            'testX': testX, 'testY': testY, 'testXTE': testXTE,
        },
        'stats': { 'mean': mean, 'std': std },
        'locations': locations, 
        'n_nodes': num_nodes,
        'ground_truth_adj': None 
    }
    return dataset_pack

# === 修改: _load_synthetic_dataset ===
def _load_synthetic_dataset(config, log_f):
    """
    加载/生成合成数据集。
    """
    P, Q = config.data.input_len, config.data.output_len
    
    # 1. 
    if config.synthetic.num_steps_after_burn_in < 1000: 
        min_steps = 2000 
        log_string(log_f, f"Warning: num_steps_after_burn_in < 1000. Re-setting to {min_steps} steps for split.")
        config.synthetic.num_steps_after_burn_in = min_steps

    # 2. 
    full_data, ground_truth_adj, locations = generate_synthetic_data( 
        config.synthetic,
        seed=config.training.seed
    )
    
    num_step, num_nodes, _ = full_data.shape
    log_string(log_f, f'Shape of generated data (after burn-in): {full_data.shape}')
    log_string(log_f, f'Shape of generated locations: {locations.shape}')
    
    # 3. 
    train_steps = round(config.data.train_ratio * num_step)
    val_steps = round(config.data.val_ratio * num_step)
    test_steps = num_step - train_steps - val_steps
    
    # 
    if train_steps < P + Q or val_steps < P + Q or test_steps < P + Q:
        log_string(log_f, f"Error: Not enough data for split after burn-in. (Need {P+Q} steps per split)")
        log_string(log_f, f"Train: {train_steps}, Val: {val_steps}, Test: {test_steps}")
        raise ValueError("Not enough data. Increase num_steps_after_burn_in in config.")

    trainData = full_data[: train_steps]
    valData = full_data[train_steps : train_steps + val_steps]
    testData = full_data[-test_steps :]

    # 4. 
    trainX, trainY = seq2instance(trainData, P, Q)
    valX, valY = seq2instance(valData, P, Q)
    testX, testY = seq2instance(testData, P, Q)
    
    # 5. 
    tod = config.model.tod_size
    dow = config.model.dow_size
    TE = np.zeros([num_step, 2])
    TE[:,0] = np.array([i % tod for i in range(num_step)])
    TE[:,1] = np.array([(i // tod) % dow for i in range(num_step)])
    TE_tile = np.repeat(np.expand_dims(TE, 1), num_nodes, 1)
    
    trainXTE, _ = seq2instance(TE_tile[: train_steps], P, Q)
    valXTE, _ = seq2instance(TE_tile[train_steps : train_steps + val_steps], P, Q)
    testXTE, _ = seq2instance(TE_tile[-test_steps :], P, Q)
    
    mean, std = np.mean(trainX), np.std(trainX)

    log_string(log_f, "Data shapes (Samples, Time, Nodes, Features):")
    log_string(log_f, f'Shape of Train X (P={P}): {trainX.shape}')
    log_string(log_f, f'Shape of Train Y (Q={Q}): {trainY.shape}')
    log_string(log_f, f'Shape of Validation X (P={P}): {valX.shape}')
    log_string(log_f, f'Shape of Validation Y (Q={Q}): {valY.shape}')
    log_string(log_f, f'Shape of Test Y (P={P}): {testX.shape}')
    log_string(log_f, f'Shape of Test Y (Q={Q}): {testY.shape}')
    log_string(log_f, f'Mean (Train X): {mean} & Std (Train X): {std}')

    dataset_pack = {
        'data': {
            'trainX': trainX, 'trainY': trainY, 'trainXTE': trainXTE,
            'valX': valX, 'valY': valY, 'valXTE': valXTE,
            'testX': testX, 'testY': testY, 'testXTE': testXTE,
        },
        'stats': { 'mean': mean, 'std': std }, # 
        'locations': locations, 
        'n_nodes': num_nodes,
        'ground_truth_adj': ground_truth_adj
    }
    return dataset_pack

# === 主调度函数 (无变化) ===
def load_dataset(config, log_f):
    """
    根据配置加载数据集 (真实世界或合成)。
    """
    if config.data.dataset_type == "real_world":
        log_string(log_f, "Loading Real-World Spatio-Temporal Dataset...")
        # 
        return _load_real_world_dataset(config.data, config.model, log_f) 
    elif config.data.dataset_type == "synthetic":
        log_string(log_f, "Generating Synthetic Causal Dataset...")
        # 
        return _load_synthetic_dataset(config, log_f)
    else:
        raise ValueError(f"Unknown dataset_type: {config.data.dataset_type}")