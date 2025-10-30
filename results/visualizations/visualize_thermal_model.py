#!/usr/bin/env python3
"""
Visualization script to understand thermal model performance
Run this to see what the model is actually learning
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.thermal_model import ThermalAnomalyDetector
from src.thermal_data_loader import get_robust_thermal_dataloaders
from src.metrics import compute_metrics

def load_model(checkpoint_path='models/checkpoints/thermal_best.pt', device='cuda'):
    """Load trained model"""
    model = ThermalAnomalyDetector(mode='direct').to(device)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from {checkpoint_path}")
        print(f"  Trained for {checkpoint.get('epoch', '?')} epochs")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
        print("   Using randomly initialized model")
    
    model.eval()
    return model

def visualize_predictions(model, data_loader, num_samples=5, device='cuda'):
    """Visualize model predictions vs ground truth"""
    
    print("\n" + "="*60)
    print("PREDICTION VISUALIZATION")
    print("="*60)
    
    samples_shown = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if samples_shown >= num_samples:
                break
                
            thermal = batch['thermal'].to(device).float()
            gt = batch['gt'].cpu().numpy()
            names = batch['name']
            
            # Get predictions
            pred_logits = model(thermal)
            pred = torch.sigmoid(pred_logits).cpu().numpy()
            
            # Process each sample in batch
            for i in range(thermal.shape[0]):
                if samples_shown >= num_samples:
                    break
                
                # Extract data
                thermal_img = thermal[i, 0].cpu().numpy()
                pred_img = pred[i, 0]
                gt_img = gt[i]
                name = names[i]
                
                # Compute metrics for this sample
                metrics = compute_metrics(pred_img.flatten(), gt_img.flatten())
                
                # Find optimal threshold
                best_f1 = 0
                best_thresh = 0.5
                for thresh in np.linspace(0.1, 0.9, 17):
                    pred_binary = (pred_img > thresh).astype(float)
                    m = compute_metrics(pred_binary.flatten(), gt_img.flatten())
                    if m['f1'] > best_f1:
                        best_f1 = m['f1']
                        best_thresh = thresh
                
                # Create visualization
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'Sample: {name}\nF1={metrics["f1"]:.3f} (thresh=0.5) | Best F1={best_f1:.3f} (thresh={best_thresh:.2f})', 
                           fontsize=14, fontweight='bold')
                
                # Row 1: Input, Prediction, Ground Truth
                axes[0, 0].imshow(thermal_img, cmap='hot')
                axes[0, 0].set_title('Input Thermal Image')
                axes[0, 0].axis('off')
                
                im1 = axes[0, 1].imshow(pred_img, cmap='viridis', vmin=0, vmax=1)
                axes[0, 1].set_title(f'Prediction (continuous)')
                axes[0, 1].axis('off')
                plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
                
                axes[0, 2].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
                axes[0, 2].set_title('Ground Truth')
                axes[0, 2].axis('off')
                
                # Row 2: Binary predictions at different thresholds
                pred_05 = (pred_img > 0.5).astype(float)
                pred_opt = (pred_img > best_thresh).astype(float)
                
                axes[1, 0].imshow(pred_05, cmap='gray', vmin=0, vmax=1)
                axes[1, 0].set_title(f'Pred @ thresh=0.5\nF1={metrics["f1"]:.3f}')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(pred_opt, cmap='gray', vmin=0, vmax=1)
                axes[1, 1].set_title(f'Pred @ thresh={best_thresh:.2f}\nF1={best_f1:.3f}')
                axes[1, 1].axis('off')
                
                # Error map
                error = np.abs(gt_img - pred_img)
                im2 = axes[1, 2].imshow(error, cmap='RdYlBu_r', vmin=0, vmax=1)
                axes[1, 2].set_title('Prediction Error')
                axes[1, 2].axis('off')
                plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
                
                plt.tight_layout()
                
                # Save figure
                output_dir = Path('results/visualizations')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'thermal_pred_sample_{samples_shown:02d}.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"✓ Saved visualization to {output_path}")
                
                plt.close()
                
                # Print sample statistics
                print(f"\nSample {samples_shown + 1}: {name}")
                print(f"  Thermal range: [{thermal_img.min():.3f}, {thermal_img.max():.3f}]")
                print(f"  Prediction range: [{pred_img.min():.3f}, {pred_img.max():.3f}]")
                print(f"  GT positive pixels: {gt_img.sum():.0f} ({100*gt_img.mean():.2f}%)")
                print(f"  Pred positive @ 0.5: {pred_05.sum():.0f} ({100*pred_05.mean():.2f}%)")
                print(f"  Pred positive @ {best_thresh:.2f}: {pred_opt.sum():.0f} ({100*pred_opt.mean():.2f}%)")
                print(f"  F1 @ 0.5: {metrics['f1']:.3f}")
                print(f"  F1 @ {best_thresh:.2f}: {best_f1:.3f} (↑{100*(best_f1-metrics['f1'])/max(metrics['f1'],0.001):.1f}%)")
                print(f"  PR-AUC: {metrics['pr_auc']:.3f}")
                print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
                
                samples_shown += 1
    
    print(f"\n✓ Visualized {samples_shown} samples")
    print(f"  Saved to: results/visualizations/")

def analyze_threshold_sensitivity(model, data_loader, device='cuda'):
    """Analyze how F1 changes with threshold"""
    
    print("\n" + "="*60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*60)
    
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for batch in data_loader:
            thermal = batch['thermal'].to(device).float()
            gt = batch['gt'].cpu().numpy()
            
            pred = torch.sigmoid(model(thermal))
            pred = pred.squeeze(1).cpu().numpy()
            
            all_preds.append(pred)
            all_gts.append(gt)
    
    all_preds = np.concatenate(all_preds).flatten()
    all_gts = np.concatenate(all_gts).flatten()
    
    # Test different thresholds
    thresholds = np.linspace(0.05, 0.95, 19)
    f1_scores = []
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        pred_binary = (all_preds > thresh).astype(float)
        metrics = compute_metrics(pred_binary, all_gts)
        f1_scores.append(metrics['f1'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    
    # Find optimal threshold
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\n📊 Threshold Analysis Results:")
    print(f"  Optimal threshold: {best_thresh:.2f}")
    print(f"  F1 @ optimal: {best_f1:.3f}")
    print(f"  F1 @ 0.5: {f1_scores[10]:.3f} (standard)")
    print(f"  Improvement: +{100*(best_f1-f1_scores[10])/max(f1_scores[10],0.001):.1f}%")
    
    # Plot threshold curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 vs Threshold
    axes[0].plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
    axes[0].axvline(best_thresh, color='r', linestyle='--', label=f'Optimal ({best_thresh:.2f})')
    axes[0].axvline(0.5, color='gray', linestyle=':', label='Standard (0.5)')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall tradeoff
    axes[1].plot(thresholds, precisions, 'g-', linewidth=2, label='Precision')
    axes[1].plot(thresholds, recalls, 'orange', linewidth=2, label='Recall')
    axes[1].axvline(best_thresh, color='r', linestyle='--', label=f'Optimal ({best_thresh:.2f})')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision-Recall Tradeoff')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path('results/visualizations/threshold_analysis.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved threshold analysis to {output_path}")
    plt.close()
    
    return best_thresh, best_f1

def main():
    """Main analysis function"""
    
    print("\n" + "="*60)
    print("THERMAL MODEL ANALYSIS & VISUALIZATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load model
    model = load_model(device=device)
    
    # Load data
    print("\nLoading validation data...")
    _, val_loader = get_robust_thermal_dataloaders(
        batch_size=4,
        data_dir='./data',
        target_size=(256, 256)
    )
    
    if not val_loader:
        print("ERROR: No validation data found!")
        return
    
    print(f"✓ Loaded {len(val_loader)} validation batches")
    
    # Analyze threshold sensitivity
    best_thresh, best_f1 = analyze_threshold_sensitivity(model, val_loader, device)
    
    # Visualize predictions
    visualize_predictions(model, val_loader, num_samples=5, device=device)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Best F1 Score: {best_f1:.3f} @ threshold={best_thresh:.2f}")
    print(f"✓ Visualizations saved to: results/visualizations/")
    print(f"\n💡 Key Insights:")
    print(f"   • If best_thresh << 0.5: Model is too conservative (underdetecting)")
    print(f"   • If best_thresh >> 0.5: Model is too aggressive (overdetecting)")
    print(f"   • Large F1 improvement with threshold tuning → good model, wrong threshold")
    print(f"   • Small F1 improvement → model needs more training")

if __name__ == '__main__':
    main()