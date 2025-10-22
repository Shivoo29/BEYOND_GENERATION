"""
Utility functions for visualization, export, and competition submission
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import rasterio
from rasterio.transform import from_bounds
import pandas as pd
from pathlib import Path
import hashlib
from typing import Dict, Tuple, Optional, Union
import seaborn as sns
from datetime import datetime

def compute_file_hash(filepath: str) -> str:
    """
    Compute SHA-256 hash of a file
    Required for competition submission
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_geotiff(data: np.ndarray, 
                 output_path: str,
                 crs: str = 'EPSG:4326',
                 bounds: Optional[Tuple] = None) -> None:
    """
    Save anomaly detection results as GeoTIFF
    
    Args:
        data: Anomaly map (H, W)
        output_path: Output file path
        crs: Coordinate reference system
        bounds: (left, bottom, right, top) in CRS units
    """
    height, width = data.shape
    
    # Create transform
    if bounds:
        transform = from_bounds(*bounds, width, height)
    else:
        # Default transform (pixel coordinates)
        transform = from_bounds(0, 0, width, height, width, height)
    
    # Save as GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)


def visualize_hyperspectral(hsi: np.ndarray, 
                           bands: Tuple[int, int, int] = None,
                           title: str = "Hyperspectral Image") -> np.ndarray:
    """
    Create RGB visualization from hyperspectral image
    
    Args:
        hsi: Hyperspectral image (H, W, B)
        bands: RGB band indices (default: automatic selection)
        title: Plot title
        
    Returns:
        RGB image (H, W, 3)
    """
    if bands is None:
        # Automatic band selection for common sensors
        n_bands = hsi.shape[2]
        if n_bands >= 200:  # Likely AVIRIS or similar
            bands = (50, 30, 20)  # Near-IR, Red, Green
        elif n_bands >= 100:
            bands = (n_bands//2, n_bands//3, n_bands//4)
        else:
            bands = (min(29, n_bands-1), min(19, n_bands-1), min(9, n_bands-1))
    
    # Extract RGB bands
    rgb = np.stack([hsi[:, :, b] for b in bands], axis=2)
    
    # Normalize for visualization
    for i in range(3):
        band = rgb[:, :, i]
        p2, p98 = np.percentile(band, [2, 98])
        rgb[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    return rgb


def plot_detection_results(hsi: np.ndarray,
                          gt: np.ndarray,
                          pred: np.ndarray,
                          scores: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Comprehensive visualization of anomaly detection results
    
    Args:
        hsi: Hyperspectral image (H, W, B)
        gt: Ground truth (H, W)
        pred: Binary predictions (H, W)
        scores: Anomaly scores (H, W)
        save_path: Path to save figure
    """
    # Create RGB visualization
    rgb = visualize_hyperspectral(hsi)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. RGB composite
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Composite')
    axes[0, 0].axis('off')
    
    # 2. Ground truth
    axes[0, 1].imshow(gt, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # 3. Predictions
    axes[0, 2].imshow(pred, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Predictions')
    axes[0, 2].axis('off')
    
    # 4. Overlay on RGB
    overlay = rgb.copy()
    overlay[pred > 0.5] = [1, 0, 0]  # Red for anomalies
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Detection Overlay')
    axes[1, 0].axis('off')
    
    # 5. Anomaly scores (if provided)
    if scores is not None:
        im = axes[1, 1].imshow(scores, cmap='viridis')
        axes[1, 1].set_title('Anomaly Scores')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    else:
        axes[1, 1].axis('off')
    
    # 6. Error map
    error = np.abs(gt - pred)
    axes[1, 2].imshow(error, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1, 2].set_title('Error Map')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_submission_report(results: Dict[str, np.ndarray],
                            metrics: Dict[str, float],
                            output_dir: str,
                            team_id: str = "TEAM_EBT") -> Dict[str, str]:
    """
    Create complete submission package for competition
    
    Args:
        results: Dictionary with 'anomaly_map', 'hsi', 'thermal' arrays
        metrics: Dictionary with evaluation metrics
        output_dir: Directory to save submission files
        team_id: Team identifier for file naming
        
    Returns:
        Dictionary with paths to all submission files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_files = {}
    
    # 1. Save anomaly map as GeoTIFF
    geotiff_path = output_dir / f"{team_id}_{timestamp}_anomaly.tif"
    save_geotiff(results['anomaly_map'], str(geotiff_path))
    submission_files['geotiff'] = str(geotiff_path)
    
    # 2. Save visualization as PNG
    png_path = output_dir / f"{team_id}_{timestamp}_visualization.png"
    
    if 'hsi' in results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RGB composite
        rgb = visualize_hyperspectral(results['hsi'])
        axes[0].imshow(rgb)
        axes[0].set_title('Input HSI (RGB)')
        axes[0].axis('off')
        
        # Anomaly map
        axes[1].imshow(results['anomaly_map'], cmap='hot')
        axes[1].set_title('Anomaly Detection')
        axes[1].axis('off')
        
        # Overlay
        overlay = rgb.copy()
        overlay[results['anomaly_map'] > 0.5] = [1, 0, 0]
        axes[2].imshow(overlay)
        axes[2].set_title('Detection Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # Simple anomaly map visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(results['anomaly_map'], cmap='hot')
        plt.title('Anomaly Detection Results')
        plt.colorbar()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    submission_files['png'] = str(png_path)
    
    # 3. Create Excel report
    excel_path = output_dir / f"{team_id}_{timestamp}_report.xlsx"
    
    with pd.ExcelWriter(excel_path) as writer:
        # Metrics sheet
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
        # Configuration sheet
        config_data = {
            'Team ID': [team_id],
            'Timestamp': [timestamp],
            'Model': ['Energy-Based Transformer'],
            'Image Size': [f"{results['anomaly_map'].shape[0]}x{results['anomaly_map'].shape[1]}"],
            'Processing Time (s)': [metrics.get('processing_time', 'N/A')],
            'Hardware': [get_hardware_info()]
        }
        config_df = pd.DataFrame(config_data)
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
        
        # Model hash sheet
        model_hash_data = {
            'Model File': ['model_checkpoint.pt'],
            'SHA-256 Hash': [metrics.get('model_hash', 'N/A')]
        }
        hash_df = pd.DataFrame(model_hash_data)
        hash_df.to_excel(writer, sheet_name='Model Hash', index=False)
    
    submission_files['excel'] = str(excel_path)
    
    # 4. Create submission summary
    summary_path = output_dir / f"{team_id}_{timestamp}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Team ID: {team_id}\n")
        f.write(f"Submission Time: {timestamp}\n")
        f.write(f"Model: Energy-Based Transformer (EBT)\n")
        f.write(f"\nMetrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write(f"\nSubmission Files:\n")
        for key, path in submission_files.items():
            f.write(f"  {key}: {path}\n")
    
    submission_files['summary'] = str(summary_path)
    
    print(f"Submission package created in {output_dir}")
    return submission_files


def get_hardware_info() -> str:
    """Get hardware information for submission report"""
    import platform
    info = []
    
    # CPU info
    info.append(f"CPU: {platform.processor()}")
    
    # GPU info
    if torch.cuda.is_available():
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        info.append(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        info.append("GPU: Not available")
    
    # System info
    info.append(f"OS: {platform.system()} {platform.release()}")
    
    return " | ".join(info)


def plot_training_history(history: Dict[str, list], save_path: Optional[str] = None) -> None:
    """
    Plot training history with loss and metrics
    
    Args:
        history: Dictionary with 'loss', 'val_loss', 'metrics' etc.
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    if 'loss' in history:
        axes[0, 0].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # F1 Score
    if 'f1' in history:
        axes[0, 1].plot(history['f1'], label='Train F1')
        if 'val_f1' in history:
            axes[0, 1].plot(history['val_f1'], label='Val F1')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # PR-AUC
    if 'pr_auc' in history:
        axes[1, 0].plot(history['pr_auc'], label='Train PR-AUC')
        if 'val_pr_auc' in history:
            axes[1, 0].plot(history['val_pr_auc'], label='Val PR-AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PR-AUC')
        axes[1, 0].set_title('PR-AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning Rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_energy_landscape(model: torch.nn.Module,
                              X: torch.Tensor,
                              S: torch.Tensor,
                              grid_size: int = 50,
                              device: str = 'cuda') -> np.ndarray:
    """
    Visualize the energy landscape learned by the model
    Useful for understanding what the model has learned
    
    Args:
        model: Trained EBT model
        X: Hyperspectral input (1, H, W, C)
        S: Sparse component (1, H, W, C)
        grid_size: Resolution of visualization grid
        device: Device for computation
        
    Returns:
        Energy landscape array
    """
    model.eval()
    H, W = X.shape[1:3]
    
    # Create grid of anomaly maps
    energies = np.zeros((grid_size, grid_size))
    
    with torch.no_grad():
        for i, density in enumerate(np.linspace(0, 1, grid_size)):
            for j, smoothness in enumerate(np.linspace(0, 1, grid_size)):
                # Create synthetic anomaly map
                A = torch.zeros(1, H, W).to(device)
                
                # Add random anomalies with given density
                mask = torch.rand(H, W) < density
                A[0, mask] = 1
                
                # Apply smoothing based on smoothness parameter
                if smoothness > 0:
                    kernel_size = int(smoothness * 10) * 2 + 1
                    A = torch.nn.functional.avg_pool2d(
                        A.unsqueeze(1), 
                        kernel_size, 
                        stride=1, 
                        padding=kernel_size//2
                    ).squeeze(1)
                
                # Compute energy
                energy = model(X, S, A)
                energies[i, j] = energy.item()
    
    return energies


def analyze_false_positives(gt: np.ndarray, 
                           pred: np.ndarray,
                           hsi: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze false positive patterns
    
    Args:
        gt: Ground truth
        pred: Predictions
        hsi: Optional hyperspectral image for spectral analysis
        
    Returns:
        Dictionary with analysis results
    """
    fp_mask = (pred > 0.5) & (gt == 0)
    tp_mask = (pred > 0.5) & (gt == 1)
    
    analysis = {
        'num_false_positives': fp_mask.sum(),
        'num_true_positives': tp_mask.sum(),
        'false_positive_rate': fp_mask.sum() / (gt == 0).sum() if (gt == 0).sum() > 0 else 0
    }
    
    if hsi is not None and fp_mask.sum() > 0:
        # Analyze spectral characteristics
        fp_spectra = hsi[fp_mask].mean(axis=0)
        bg_spectra = hsi[gt == 0].mean(axis=0)
        
        analysis['fp_spectral_deviation'] = np.linalg.norm(fp_spectra - bg_spectra)
        analysis['fp_mean_intensity'] = hsi[fp_mask].mean()
        analysis['bg_mean_intensity'] = hsi[gt == 0].mean()
    
    return analysis


# Export all functions
__all__ = [
    'compute_file_hash',
    'save_geotiff',
    'visualize_hyperspectral',
    'plot_detection_results',
    'create_submission_report',
    'get_hardware_info',
    'plot_training_history',
    'visualize_energy_landscape',
    'analyze_false_positives'
]