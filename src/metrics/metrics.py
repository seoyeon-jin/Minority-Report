"""평가 메트릭 함수들"""
import numpy as np
import torch
from scipy.stats import pearsonr
from dtaidistance import dtw


def masked_mae(pred, target, mask=None):
    """
    Masked Mean Absolute Error
    
    Args:
        pred: (B, T, D) or (B, T) - 예측값
        target: (B, T, D) or (B, T) - 실제값
        mask: (B, T) - 마스크 (1=유효, 0=무효). None이면 전체 사용
    
    Returns:
        float: MAE 값
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    diff = np.abs(pred - target)
    
    if mask is not None:
        # mask를 dimension에 맞게 확장
        if len(diff.shape) == 3 and len(mask.shape) == 2:
            mask = mask[..., np.newaxis]  # (B, T, 1)
        
        mask = mask.astype(bool)
        if mask.sum() == 0:
            return np.nan
        return diff[mask].mean()
    else:
        return diff.mean()


def masked_dtw_distance(pred, target, mask=None):
    """
    Masked Dynamic Time Warping Distance
    
    Args:
        pred: (B, T, D) or (B, T) - 예측값
        target: (B, T, D) or (B, T) - 실제값
        mask: (B, T) - 마스크. None이면 전체 사용
    
    Returns:
        float: 평균 DTW 거리
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 3D -> 2D로 변환 (평균)
    if len(pred.shape) == 3:
        pred = pred.mean(axis=-1)  # (B, T)
    if len(target.shape) == 3:
        target = target.mean(axis=-1)  # (B, T)
    
    batch_size = pred.shape[0]
    dtw_distances = []
    
    for i in range(batch_size):
        pred_seq = pred[i]
        target_seq = target[i]
        
        if mask is not None:
            valid_mask = mask[i].astype(bool)
            if valid_mask.sum() < 2:
                continue
            pred_seq = pred_seq[valid_mask]
            target_seq = target_seq[valid_mask]
        
        # DTW 계산
        try:
            distance = dtw.distance(pred_seq, target_seq)
            dtw_distances.append(distance)
        except:
            continue
    
    if len(dtw_distances) == 0:
        return np.nan
    return np.mean(dtw_distances)


def masked_pearson(pred, target, mask=None):
    """
    Masked Pearson Correlation Coefficient
    
    Args:
        pred: (B, T, D) or (B, T) - 예측값
        target: (B, T, D) or (B, T) - 실제값
        mask: (B, T) - 마스크. None이면 전체 사용
    
    Returns:
        float: Pearson 상관계수
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Flatten
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    if mask is not None:
        # mask를 dimension에 맞게 확장
        if len(pred.shape) == 3 and len(mask.shape) == 2:
            mask_expanded = np.repeat(mask[..., np.newaxis], pred.shape[-1], axis=-1)
        else:
            mask_expanded = mask
        
        mask_flat = mask_expanded.flatten().astype(bool)
        
        if mask_flat.sum() < 2:
            return np.nan
        
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]
    
    # Pearson 계산
    try:
        corr, _ = pearsonr(pred_flat, target_flat)
        return corr
    except:
        return np.nan


def compute_all_metrics(pred, target, mask=None):
    """
    모든 메트릭 계산
    
    Returns:
        dict: {'mae': float, 'dtw': float, 'pearson': float}
    """
    metrics = {
        'mae': masked_mae(pred, target, mask),
        'dtw': masked_dtw_distance(pred, target, mask),
        'pearson': masked_pearson(pred, target, mask)
    }
    return metrics

