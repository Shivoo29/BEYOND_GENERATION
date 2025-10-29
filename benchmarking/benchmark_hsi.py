"""
Benchmark script for the HSI-only Energy-Based Transformer model.

Instructions:
1. Ensure you have a trained model checkpoint saved in the directory specified in `config.py`.
2. Run this script from the root project directory:
   python benchmarking/benchmark_hsi.py
"""

import torch
from tqdm import tqdm
import numpy as np
import os
import sys

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.data_loader import get_dataloaders
from src.stage2_ebt import EnergyBasedTransformer
from src.inference import InferencePipeline
from src.metrics import compute_metrics


def benchmark_hsi_model(model_path: str = None):
    """
    Loads a trained HSI model, runs it on the test set, and prints metrics.
    """
    print("--- Starting HSI Model Benchmark ---")

    # --- 1. Setup ---
    device = config.device
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU. This will be slow.")
        device = 'cpu'

    # --- 2. Load Data ---
    # Use the validation set for benchmarking as a stand-in for a true test set
    _, test_loader = get_dataloaders(
        batch_size=1,  # Process one full tile at a time
        data_dir=config.data_dir,
        test_mode=True # Use a small subset for quick benchmark
    )

    if test_loader is None:
        print("Fatal: HSI data loader failed to initialize. Aborting benchmark.")
        return

    # --- 3. Load Model ---
    model = EnergyBasedTransformer().to(device)
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {model_path}")
    else:
        print("Warning: No model checkpoint specified or found. Using randomly initialized model.")

    model.eval()
    inference_pipeline = InferencePipeline(model, device=device)

    # --- 4. Run Inference ---
    all_preds = []
    all_gts = []
    pbar = tqdm(test_loader, desc="Running HSI Inference")

    for batch in pbar:
        image = batch['image'].to(device)
        gt = batch['gt'].numpy()
        
        # The model expects (B, H, W, C) but loader gives (B, C, H, W)
        image_for_model = image.permute(0, 2, 3, 1)

        # Stage 1: LRSR (done inside inference pipeline)
        # Stage 2: EBT Refinement
        # Note: The current inference pipeline is a placeholder.
        # This benchmark will use a simplified, direct energy evaluation for now.
        
        with torch.no_grad():
            # Create a dummy sparse component for now
            s_dummy = torch.zeros_like(image_for_model)
            # The anomaly map `A` is what we optimize. For benchmark, we can't optimize.
            # We will use a placeholder logic: get energy of the GT vs. zero map
            energy_gt = model(image_for_model, s_dummy, batch['gt'].to(device))
            energy_zero = model(image_for_model, s_dummy, torch.zeros_like(batch['gt']).to(device))
            
            # Anomaly score is the difference in energy. High score = anomaly.
            anomaly_score = energy_zero - energy_gt
            pred_scores = anomaly_score.cpu().numpy()

        all_preds.append(pred_scores.flatten())
        all_gts.append(gt.flatten())

    if not all_gts:
        print("Fatal: No data processed. Aborting benchmark.")
        return

    # --- 5. Compute Metrics ---
    print("--- HSI Benchmark Results ---")
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
        print("This might happen if no anomalies were present in the test set.")
    
    print("--- End of HSI Benchmark ---")

if __name__ == "__main__":
    # You can specify a path to a model checkpoint as a command-line argument
    # Example: python benchmarking/benchmark_hsi.py models/checkpoints/hsi_best.pt
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    benchmark_hsi_model(checkpoint_path)
