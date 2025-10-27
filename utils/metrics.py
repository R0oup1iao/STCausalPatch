# STCP/utils/metrics.py
# (已修复 Bug 2: 损失函数错误)

import torch
import numpy as np

# === 代码直接复制自: ===
# lmissher/patchstg/PatchSTG-feb68c369ac51fee2e730c2d27393bb9103c8a8c/lib/utils.py
# (但已为 STCP 修复)

def metric(pred, label, null_val=0.0):
    """
    计算 MAE, RMSE, MAPE。
    #!#!#! 修复: 添加 null_val 参数。
    """
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        
        #!#!#! 修复: 检查 null_val 是否为 None
        if null_val is None or np.isnan(null_val):
            mask = np.ones_like(label, dtype=np.float32)
        else:
            mask = np.not_equal(label, null_val).astype(np.float32)
        
        mask_mean = np.mean(mask)
        if mask_mean == 0:
            # 
            mae, rmse, mape = 0.0, 0.0, 0.0
            return mae, rmse, mape

        #!#!#! 修复: 移动 mask /= np.mean(mask) 
        # 
        
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        #!#!#! 修复: 只有在 mask_mean > 0 时才进行归一化
        if mask_mean > 0:
            mae = mae / mask_mean 

        rmse = np.nan_to_num(rmse * mask)
        rmse = np.mean(rmse)
        if mask_mean > 0:
            rmse = np.sqrt(rmse / mask_mean)
            
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        if mask_mean > 0:
            mape = mape / mask_mean
            
    return mae, rmse, mape

def masked_mae(preds, labels, null_val=np.nan):
    """
    Masked MAE 损失函数。
    #!#!#! 修复: 彻底重写此函数，使其更健壮。
    """
    
    if null_val is None or torch.isnan(null_val):
        # 
        mask = torch.ones_like(labels, dtype=torch.float32)
    else:
        # 
        mask = (labels != null_val).float()

    mask_mean = torch.mean(mask)
    
    if mask_mean == 0:
        # 
        return torch.tensor(0.0, device=preds.device)
    
    loss = torch.abs(preds - labels)
    loss = loss * mask
    
    # 
    loss = torch.sum(loss) / torch.sum(mask)
    
    return loss