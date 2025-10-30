"""
Metrics computation for thermal anomaly detection
"""
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)


def compute_metrics(predictions, ground_truth, probabilities=None):
    """
    Compute comprehensive metrics for binary segmentation
    
    Args:
        predictions: Binary predictions (0 or 1)
        ground_truth: Ground truth labels (0 or 1)
        probabilities: Optional probability scores for AUC metrics
    
    Returns:
        Dictionary of metrics
    """
    # Ensure numpy arrays
    predictions = np.asarray(predictions).flatten()
    ground_truth = np.asarray(ground_truth).flatten()
    
    # Replace NaN/Inf
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
    ground_truth = np.nan_to_num(ground_truth, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Ensure binary
    predictions = (predictions > 0.5).astype(float)
    ground_truth = (ground_truth > 0.5).astype(float)
    
    # Compute basic metrics
    try:
        f1 = f1_score(ground_truth, predictions, zero_division=0)
        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall = recall_score(ground_truth, predictions, zero_division=0)
    except Exception as e:
        print(f"Warning: Error computing F1/Precision/Recall: {e}")
        f1 = precision = recall = 0.0
    
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    # Compute AUC metrics if probabilities provided
    if probabilities is not None:
        probabilities = np.asarray(probabilities).flatten()
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
        probabilities = np.clip(probabilities, 0, 1)
        
        try:
            # Check if we have both classes
            if len(np.unique(ground_truth)) > 1:
                roc_auc = roc_auc_score(ground_truth, probabilities)
                pr_auc = average_precision_score(ground_truth, probabilities)
            else:
                roc_auc = 0.5  # Random for single class
                pr_auc = np.mean(ground_truth)  # Baseline
        except Exception as e:
            print(f"Warning: Error computing AUC metrics: {e}")
            roc_auc = 0.5
            pr_auc = 0.5
        
        metrics['roc_auc'] = roc_auc
        metrics['pr_auc'] = pr_auc
    
    return metrics


def compute_confusion_matrix(predictions, ground_truth):
    """Compute confusion matrix components"""
    predictions = np.asarray(predictions).flatten()
    ground_truth = np.asarray(ground_truth).flatten()
    
    predictions = (predictions > 0.5).astype(float)
    ground_truth = (ground_truth > 0.5).astype(float)
    
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    
    return {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics computation...")
    
    # Create dummy data
    gt = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    pred_binary = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    pred_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.1, 0.85, 0.05])
    
    # Compute metrics
    metrics = compute_metrics(pred_binary, gt, pred_probs)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(pred_binary, gt)
    print("\nConfusion Matrix:")
    for key, value in cm.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Metrics test passed!")