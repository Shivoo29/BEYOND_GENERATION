'''
Benchmark script for the HSI-Thermal Fusion model.

Instructions:
1. This script assumes an aligned HSI-thermal dataset exists.
   You will need to create a new DataLoader for this.
2. Ensure you have a trained fusion model checkpoint.
3. Run this script from the root project directory:
   python benchmarking/benchmark_fusion.py
'''


import torch
from tqdm import tqdm
import numpy as np
import os
import sys

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
# NOTE: We need a new data loader for aligned HSI-Thermal data.
# from src.fusion_data_loader import get_fusion_dataloaders
from src.multimodel_fusion import MultimodalEnergyFunction, MultimodalInference
from src.metrics import compute_metrics

# Placeholder for the actual fusion data loader
def get_fusion_dataloaders(batch_size, data_dir):
    print("\nWARNING: This is a placeholder for the real fusion data loader.")
    print("You must implement a Dataset that returns aligned HSI and Thermal images.")
    # Returning None to prevent execution without a real implementation
    return None, None

def benchmark_fusion_model(model_path: str = None):
    """
    Loads a trained fusion model, runs it on the test set, and prints metrics.
    """
    print("--- Starting HSI-Thermal Fusion Model Benchmark ---")

    # --- 1. Setup ---
    device = config.device
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
        device = 'cpu'

    # --- 2. Load Data ---
    # This is a placeholder! You need to implement a loader that yields
    # batches with both 'hsi_image' and 'thermal_image'.
    _, test_loader = get_fusion_dataloaders(batch_size=1, data_dir=config.data_dir)

    if test_loader is None:
        print("Fatal: Fusion data loader not implemented. Aborting benchmark.")
        return

    # --- 3. Load Model ---
    model = MultimodalEnergyFunction().to(device)
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {model_path}")
    else:
        print("Warning: No model checkpoint specified or found. Using randomly initialized model.")

    model.eval()
    inference_pipeline = MultimodalInference(model, device=device)

    # --- 4. Run Inference ---
    all_preds = []
    all_gts = []
    pbar = tqdm(test_loader, desc="Running Fusion Inference")

    with torch.no_grad():
        for batch in pbar:
            hsi_image = batch['hsi_image'].to(device)
            thermal_image = batch['thermal_image'].to(device)
            gt = batch['gt'].numpy()
            
            # The fusion inference pipeline handles the full process
            pred_scores, _ = inference_pipeline.joint_refinement(
                hsi_image.permute(0, 2, 3, 1), # HSI to (B, H, W, C)
                None, # Sparse component is calculated inside
                thermal_image
            )
            
            all_preds.append(pred_scores.flatten())
            all_gts.append(gt.flatten())

    if not all_gts:
        print("Fatal: No data processed. Aborting benchmark.")
        return

    # --- 5. Compute Metrics ---
    print("--- Fusion Benchmark Results ---")
    try:
        full_preds = np.concatenate(all_preds)
        full_gts = np.concatenate(all_gts)
        
        metrics = compute_metrics(full_preds, full_gts)
        
        print(f"  Precision-Recall AUC (PR-AUC): {metrics['pr_auc']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  F1-Score (at 0.5 threshold): {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

    except Exception as e:
        print(f"Could not compute metrics. Error: {e}")
    
    print("--- End of Fusion Benchmark ---")

if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    benchmark_fusion_model(checkpoint_path)
