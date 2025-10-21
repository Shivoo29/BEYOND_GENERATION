import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def compute_metrics(pred, gt, threshold=0.5):
    """
    Compute all required metrics
    
    Args:
        pred: predicted anomaly scores (H, W) or flattened
        gt: ground truth binary map (H, W) or flattened
        threshold: threshold for binary predictions
        
    Returns:
        dict with all metrics
    """
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Binary predictions
    pred_binary = (pred_flat > threshold).astype(int)
    
    # Basic metrics
    tp = np.sum((pred_binary == 1) & (gt_flat == 1))
    fp = np.sum((pred_binary == 1) & (gt_flat == 0))
    fn = np.sum((pred_binary == 0) & (gt_flat == 1))
    tn = np.sum((pred_binary == 0) & (gt_flat == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # ROC-AUC
    roc_auc = roc_auc_score(gt_flat, pred_flat)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(gt_flat, pred_flat)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }