"""
Benchmark script for the Thermal-only U-Net based model.

Instructions:
1. Ensure you have a trained model checkpoint saved in the directory specified in `config.py`.
   (e.g., 'thermal_best.pt')
2. Run this script from the root project directory:
   python benchmarking/benchmark_thermal.py
"""

import torch
from tqdm import tqdm
import numpy as np
import os
import sys

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.thermal_data_loader import get_local_thermal_dataloaders, get_kaggle_thermal_dataloaders
from src.thermal_model import ThermalAnomalyDetector
from src.metrics import compute_metrics


def benchmark_thermal_model(model_path: str = None, data_source: str = 'local'):
    """
    Loads a trained thermal model, runs it on the test set, and prints metrics.
    """
    print(f"--- Starting Thermal Model Benchmark (Source: {data_source}) ---")

    # --- 1. Setup ---
    device = config.device
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
        device = 'cpu'

    # --- 2. Load Data ---
    print(f"Loading {data_source} thermal data...")
    if data_source == 'kaggle':
        _, test_loader = get_kaggle_thermal_dataloaders(batch_size=4)
    else:
        _, test_loader = get_local_thermal_dataloaders(batch_size=4, data_dir=config.data_dir)

    if test_loader is None:
        print(f"Fatal: Thermal data loader for source '{data_source}' failed. Aborting benchmark.")
        return

    # --- 3. Load Model ---
    # The thermal model is a standard U-Net, so we set mode='direct'
    model = ThermalAnomalyDetector(mode='direct').to(device)
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {model_path}")
    else:
        print("Warning: No model checkpoint specified or found. Using randomly initialized model.")

    model.eval()

    # --- 4. Run Inference ---
    all_preds = []
    all_gts = []
    pbar = tqdm(test_loader, desc="Running Thermal Inference")

    with torch.no_grad():
        for batch in pbar:
            thermal_img = batch['thermal'].to(device)
            gt = batch['gt'].numpy()
            
            # Get model prediction
            pred_scores = model(thermal_img).squeeze(1).cpu().numpy()
            
            all_preds.append(pred_scores.flatten())
            all_gts.append(gt.flatten())

    if not all_gts:
        print("Fatal: No data processed. Aborting benchmark.")
        return

    # --- 5. Compute Metrics ---
    print("--- Thermal Benchmark Results ---")
    try:
        full_preds = np.concatenate(all_preds)
        full_gts = np.concatenate(all_gts)
        
        # Check if there are any ground truth anomalies
        if np.sum(full_gts) == 0:
            print("Warning: No ground truth anomalies found in the test set.")
            print("AUC metrics are not meaningful in this case.")
        
        metrics = compute_metrics(full_preds, full_gts)
        
        print(f"  Precision-Recall AUC (PR-AUC): {metrics['pr_auc']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  F1-Score (at 0.5 threshold): {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

    except Exception as e:
        print(f"Could not compute metrics. Error: {e}")
    
    print("--- End of Thermal Benchmark ---")

if __name__ == "__main__":
    # Usage: python benchmarking/benchmark_thermal.py [path_to_model.pt] [data_source]
    # data_source can be 'local' (default) or 'kaggle'
    
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    source = sys.argv[2] if len(sys.argv) > 2 else 'local'
    
    if source not in ['local', 'kaggle']:
        print(f"Invalid data source '{source}'. Choose 'local' or 'kaggle'.")
    else:
        benchmark_thermal_model(checkpoint_path, source)
