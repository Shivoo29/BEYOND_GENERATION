#!/usr/bin/env python
"""
Test script for Thermal anomaly detection model on October 15th mock dataset
Generates competition-ready outputs with proper formatting and metrics
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import rasterio
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.thermal_model import ThermalAnomalyDetector
from src.metrics import compute_metrics
from src.utils import (
    compute_file_hash,
    save_geotiff,
    plot_detection_results
)

class ThermalTester:
    """Complete testing pipeline for Thermal model"""
    
    def __init__(self, model_path, device='cuda', target_size=(256, 256)):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.target_size = target_size
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = ThermalAnomalyDetector(mode='direct').to(self.device)
        self.load_model(model_path)
        self.model.eval()
        
        print("Thermal Model loaded successfully!")
    
    def load_model(self, model_path):
        """Load trained model checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    def load_thermal_file(self, filepath):
        """
        Load thermal file (GeoTIFF or other formats)
        Returns: (thermal_data, ground_truth, metadata)
        """
        print(f"Loading thermal file: {filepath}")
        
        try:
            if filepath.endswith('.tif') or filepath.endswith('.tiff'):
                with rasterio.open(filepath) as src:
                    # Read thermal band (usually first band)
                    thermal_data = src.read(1).astype(np.float32)
                    
                    # Try to read ground truth (second band if exists)
                    gt = None
                    if src.count >= 2:
                        gt = src.read(2).astype(np.float32)
                    
                    metadata = {
                        'filename': os.path.basename(filepath),
                        'crs': src.crs,
                        'transform': src.transform,
                        'bounds': src.bounds,
                        'shape': thermal_data.shape
                    }
                    
                    print(f"Loaded thermal data, shape: {thermal_data.shape}")
                    if gt is not None:
                        print(f"Found ground truth, shape: {gt.shape}")
                    
                    return thermal_data, gt, metadata
            
            else:
                # Try loading as numpy or image
                if filepath.endswith('.npy'):
                    thermal_data = np.load(filepath)
                else:
                    thermal_data = np.array(Image.open(filepath))
                
                metadata = {
                    'filename': os.path.basename(filepath),
                    'shape': thermal_data.shape
                }
                
                return thermal_data, None, metadata
        
        except Exception as e:
            print(f"ERROR loading file: {e}")
            raise
    
    def preprocess_thermal(self, thermal_data):
        """
        Preprocess thermal data
        Handles normalization and resizing
        """
        print(f"Preprocessing thermal data, input shape: {thermal_data.shape}")
        
        # Remove outliers using percentile clipping
        p_low, p_high = np.percentile(thermal_data, [2, 98])
        thermal_data = np.clip(thermal_data, p_low, p_high)
        
        # Normalize to [0, 1]
        thermal_data = (thermal_data - thermal_data.min()) / (thermal_data.max() - thermal_data.min() + 1e-8)
        
        # Resize to target size if needed
        if self.target_size and thermal_data.shape != self.target_size:
            print(f"Resizing from {thermal_data.shape} to {self.target_size}")
            thermal_data = cv2.resize(
                thermal_data, 
                self.target_size, 
                interpolation=cv2.INTER_LINEAR
            )
        
        print(f"Final preprocessed shape: {thermal_data.shape}")
        return thermal_data.astype(np.float32)
    
    def preprocess_gt(self, gt, target_shape):
        """Preprocess ground truth to match data shape"""
        if gt is None:
            print("No ground truth provided, creating zero mask")
            return np.zeros(target_shape, dtype=np.float32)
        
        # Resize if needed
        if gt.shape != target_shape:
            print(f"Resizing GT from {gt.shape} to {target_shape}")
            gt = cv2.resize(gt, target_shape, interpolation=cv2.INTER_NEAREST)
        
        # Binarize
        gt = (gt > 0).astype(np.float32)
        return gt
    
    def run_inference(self, thermal_data):
        """
        Run inference on thermal data
        
        Args:
            thermal_data: numpy array (H, W)
            
        Returns:
            anomaly_map: numpy array (H, W)
            processing_time: float
        """
        start_time = time.time()
        
        print("Running inference...")
        
        # Convert to tensor (B, C, H, W)
        thermal_tensor = torch.from_numpy(thermal_data).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get prediction
            anomaly_scores = self.model(thermal_tensor)
            anomaly_map = anomaly_scores.squeeze().cpu().numpy()
        
        processing_time = time.time() - start_time
        print(f"Inference completed in {processing_time:.2f} seconds")
        
        return anomaly_map, processing_time
    
    def load_ground_truth(self, filepath, data_shape):
        """
        Load separate ground truth file provided by organizers on Oct 15
        
        Args:
            filepath: path to ground truth file
            data_shape: shape to match (H, W)
            
        Returns:
            gt: ground truth array (H, W)
        """
        try:
            if filepath.endswith('.tif') or filepath.endswith('.tiff'):
                with rasterio.open(filepath) as src:
                    gt = src.read(1).astype(np.float32)
                    print(f"Loaded GT from GeoTIFF, shape: {gt.shape}")
                    
            elif filepath.endswith('.npy'):
                gt = np.load(filepath)
                print(f"Loaded GT from numpy, shape: {gt.shape}")
                
            elif filepath.endswith('.png') or filepath.endswith('.jpg'):
                gt = np.array(Image.open(filepath))
                print(f"Loaded GT from image, shape: {gt.shape}")
                
            elif filepath.endswith('.mat'):
                import scipy.io
                mat_data = scipy.io.loadmat(filepath)
                # Try common GT keys
                for key in ['map', 'Map', 'gt', 'GT', 'mask', 'groundtruth']:
                    if key in mat_data:
                        gt = mat_data[key]
                        print(f"Loaded GT from MAT file key '{key}', shape: {gt.shape}")
                        break
                else:
                    raise ValueError("No GT found in MAT file")
            
            else:
                raise ValueError(f"Unsupported GT format: {filepath}")
            
            # Preprocess
            gt = self.preprocess_gt(gt, data_shape)
            return gt
            
        except Exception as e:
            print(f"WARNING: Could not load ground truth from {filepath}: {e}")
            return None
    
    def test_single_file(self, filepath, output_dir, gt_filepath=None):
        """
        Test model on a single thermal file
        
        Args:
            filepath: path to thermal file
            output_dir: directory to save results
            gt_filepath: optional separate ground truth file path
            
        Returns:
            results dictionary with metrics and paths
        """
        print("\n" + "="*60)
        print(f"Testing: {filepath}")
        print("="*60)
        
        # Load data
        thermal_data, gt_embedded, metadata = self.load_thermal_file(filepath)
        
        # Preprocess
        thermal_data = self.preprocess_thermal(thermal_data)
        
        # Load ground truth - prioritize separate GT file if provided
        if gt_filepath and os.path.exists(gt_filepath):
            print(f"Loading separate ground truth from: {gt_filepath}")
            gt = self.load_ground_truth(gt_filepath, thermal_data.shape)
        else:
            print("Using embedded ground truth (if available)")
            gt = self.preprocess_gt(gt_embedded, thermal_data.shape)
        
        # Run inference
        anomaly_map, processing_time = self.run_inference(thermal_data)
        
        # Compute metrics if ground truth available
        metrics = {}
        if gt is not None and gt.sum() > 0:
            print("\nComputing metrics...")
            metrics = compute_metrics(anomaly_map.flatten(), gt.flatten())
            
            print("\nMetrics:")
            print(f"  F1 Score:    {metrics['f1']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
        else:
            print("\nNo ground truth available - skipping metrics")
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_stem = Path(filepath).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'filename': metadata['filename'],
            'processing_time': processing_time,
            'metrics': metrics,
            'anomaly_map': anomaly_map,
            'gt': gt,
            'thermal': thermal_data,
            'metadata': metadata
        }
        
        # 1. Save anomaly map as GeoTIFF
        geotiff_path = output_dir / f"{file_stem}_thermal_anomaly.tif"
        
        # Use original CRS and transform if available
        if 'crs' in metadata and 'transform' in metadata:
            with rasterio.open(
                geotiff_path,
                'w',
                driver='GTiff',
                height=anomaly_map.shape[0],
                width=anomaly_map.shape[1],
                count=1,
                dtype=anomaly_map.dtype,
                crs=metadata['crs'],
                transform=metadata['transform'],
                compress='lzw'
            ) as dst:
                dst.write(anomaly_map, 1)
        else:
            save_geotiff(anomaly_map, str(geotiff_path))
        
        results['geotiff_path'] = str(geotiff_path)
        print(f"\n✓ Saved GeoTIFF: {geotiff_path}")
        
        # 2. Save anomaly map as PNG
        png_path = output_dir / f"{file_stem}_thermal_anomaly.png"
        anomaly_uint8 = (np.clip(anomaly_map, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(anomaly_uint8).save(png_path)
        results['png_path'] = str(png_path)
        print(f"✓ Saved PNG: {png_path}")
        
        # 3. Save visualization
        vis_path = output_dir / f"{file_stem}_thermal_visualization.png"
        self._create_thermal_visualization(
            thermal_data,
            anomaly_map,
            gt,
            str(vis_path)
        )
        results['visualization_path'] = str(vis_path)
        print(f"✓ Saved visualization: {vis_path}")
        
        return results
    
    def _create_thermal_visualization(self, thermal_data, anomaly_map, gt, save_path):
        """Create visualization for thermal results"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 4 if gt is not None and gt.sum() > 0 else 3, 
                                figsize=(16, 4))
        
        # Thermal input
        axes[0].imshow(thermal_data, cmap='hot')
        axes[0].set_title('Thermal Input')
        axes[0].axis('off')
        
        # Anomaly map
        axes[1].imshow(anomaly_map, cmap='viridis')
        axes[1].set_title('Anomaly Scores')
        axes[1].axis('off')
        
        # Binary prediction
        axes[2].imshow(anomaly_map > 0.5, cmap='Reds')
        axes[2].set_title('Binary Detection')
        axes[2].axis('off')
        
        # Ground truth (if available)
        if gt is not None and gt.sum() > 0:
            axes[3].imshow(gt, cmap='Reds')
            axes[3].set_title('Ground Truth')
            axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_directory(self, data_dir, output_dir, gt_dir=None):
        """
        Test model on all thermal files in a directory
        
        Args:
            data_dir: directory containing thermal files
            output_dir: directory to save results
            gt_dir: optional directory containing separate ground truth files
            
        Returns:
            summary_df: pandas DataFrame with results for all files
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all thermal files
        thermal_files = (list(data_dir.rglob('*.tif')) + 
                        list(data_dir.rglob('*.tiff')) +
                        list(data_dir.rglob('*.npy')))
        
        # Filter out non-thermal files if possible
        thermal_files = [f for f in thermal_files if 'thermal' in f.name.lower() 
                        or 'landsat' in f.name.lower() 
                        or 'swir' in f.name.lower()
                        or len(thermal_files) < 10]  # If few files, include all
        
        if not thermal_files:
            print(f"No thermal files found in {data_dir}")
            return None
        
        print(f"\nFound {len(thermal_files)} thermal files to process")
        
        all_results = []
        
        for filepath in thermal_files:
            try:
                # Try to find corresponding GT file
                gt_filepath = None
                if gt_dir:
                    gt_dir_path = Path(gt_dir)
                    # Try multiple naming conventions for GT files
                    stem = filepath.stem
                    possible_gt_names = [
                        f"{stem}_gt.tif",
                        f"{stem}_GT.tif",
                        f"{stem}_groundtruth.tif",
                        f"{stem}_map.tif",
                        f"Ground_truth_{stem}.tif",
                        f"{stem}.tif",  # Same name in different directory
                        f"{stem}.mat",  # MAT file GT
                    ]
                    
                    for gt_name in possible_gt_names:
                        potential_gt = gt_dir_path / gt_name
                        if potential_gt.exists():
                            gt_filepath = str(potential_gt)
                            print(f"Found GT file: {gt_filepath}")
                            break
                
                results = self.test_single_file(
                    str(filepath), 
                    output_dir / filepath.stem,
                    gt_filepath=gt_filepath
                )
                
                # Prepare summary row
                row = {
                    'filename': results['filename'],
                    'processing_time': results['processing_time'],
                    'geotiff_path': results['geotiff_path'],
                    'png_path': results['png_path']
                }
                
                # Add metrics if available
                if results['metrics']:
                    row.update({
                        'f1': results['metrics']['f1'],
                        'precision': results['metrics']['precision'],
                        'recall': results['metrics']['recall'],
                        'roc_auc': results['metrics']['roc_auc'],
                        'pr_auc': results['metrics']['pr_auc']
                    })
                
                all_results.append(row)
                
            except Exception as e:
                print(f"ERROR processing {filepath}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create summary DataFrame
        if all_results:
            summary_df = pd.DataFrame(all_results)
            
            # Save summary
            summary_path = output_dir / 'thermal_test_summary.xlsx'
            summary_df.to_excel(summary_path, index=False)
            print(f"\n✓ Saved summary: {summary_path}")
            
            # Print summary statistics
            print("\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)
            
            if 'f1' in summary_df.columns:
                print(f"Average F1:        {summary_df['f1'].mean():.4f}")
                print(f"Average Precision: {summary_df['precision'].mean():.4f}")
                print(f"Average Recall:    {summary_df['recall'].mean():.4f}")
                print(f"Average ROC-AUC:   {summary_df['roc_auc'].mean():.4f}")
                print(f"Average PR-AUC:    {summary_df['pr_auc'].mean():.4f}")
            
            print(f"Average Time:      {summary_df['processing_time'].mean():.2f}s")
            
            return summary_df
        
        return None
    
    def create_competition_submission(self, results_dict, output_dir, team_id="TEAM_EBT"):
        """
        Create competition-ready submission package
        
        Args:
            results_dict: dictionary with test results
            output_dir: directory to save submission
            team_id: team identifier
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*60)
        print("CREATING COMPETITION SUBMISSION")
        print("="*60)
        
        # 1. Compute model hash
        model_hash = "N/A"
        checkpoint_files = list(Path('models/checkpoints').glob('thermal_best.pt'))
        if checkpoint_files:
            model_hash = compute_file_hash(str(checkpoint_files[0]))
            print(f"Model hash: {model_hash}")
        
        # 2. Create Excel report
        excel_path = output_dir / f"{team_id}_{timestamp}_Thermal_report.xlsx"
        
        with pd.ExcelWriter(excel_path) as writer:
            # Metrics sheet
            if 'metrics' in results_dict and results_dict['metrics']:
                metrics_df = pd.DataFrame([results_dict['metrics']])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Configuration sheet
            config_data = {
                'Team ID': [team_id],
                'Timestamp': [timestamp],
                'Model': ['Thermal U-Net Detector'],
                'Dataset': [results_dict.get('filename', 'Unknown')],
                'Processing Time (s)': [results_dict.get('processing_time', 'N/A')],
                'Hardware': [self._get_hardware_info()],
                'Target Size': [str(self.target_size)]
            }
            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name='Configuration', index=False)
            
            # Model hash sheet
            hash_data = {
                'Model File': ['thermal_best.pt'],
                'SHA-256 Hash': [model_hash]
            }
            hash_df = pd.DataFrame(hash_data)
            hash_df.to_excel(writer, sheet_name='Model Hash', index=False)
        
        print(f"✓ Saved Excel report: {excel_path}")
        
        # 3. Create hash file
        hash_path = output_dir / f"{team_id}_{timestamp}_thermal_model_hash.txt"
        with open(hash_path, 'w') as f:
            f.write(model_hash)
        print(f"✓ Saved hash file: {hash_path}")
        
        print("\n✅ Competition submission package ready!")
        print(f"Location: {output_dir}")
    
    def _get_hardware_info(self):
        """Get hardware information"""
        import platform
        info = []
        info.append(f"CPU: {platform.processor()}")
        if torch.cuda.is_available():
            info.append(f"GPU: {torch.cuda.get_device_name(0)}")
            info.append(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            info.append("GPU: Not available")
        return " | ".join(info)


def main():
    parser = argparse.ArgumentParser(description='Test Thermal anomaly detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (e.g., models/checkpoints/thermal_best.pt)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (file or directory)')
    parser.add_argument('--gt', type=str, default=None,
                       help='Path to ground truth file (for single file) or directory (for batch)')
    parser.add_argument('--output', type=str, default='./results/thermal_test',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--target-size', type=int, nargs=2, default=[256, 256],
                       help='Target size for resizing (height width)')
    parser.add_argument('--team-id', type=str, default='TEAM_EBT',
                       help='Team identifier for submission')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ThermalTester(
        args.model, 
        device=args.device,
        target_size=tuple(args.target_size)
    )
    
    # Test data
    data_path = Path(args.data)
    
    if data_path.is_file():
        # Single file
        results = tester.test_single_file(
            str(data_path),
            args.output,
            gt_filepath=args.gt
        )
        
        # Create competition submission
        tester.create_competition_submission(
            results,
            Path(args.output) / 'submission',
            team_id=args.team_id
        )
        
    elif data_path.is_dir():
        # Directory of files
        summary_df = tester.test_directory(
            str(data_path),
            args.output,
            gt_dir=args.gt
        )
        
        if summary_df is not None and len(summary_df) > 0:
            print("\nNote: Submission packages created for each processed file")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
    print(f"Results saved to: {args.output}")
    
    if args.gt:
        print(f"\n📊 Ground truth comparison completed!")
        print(f"Ground truth source: {args.gt}")
    else:
        print(f"\n⚠️  No separate ground truth provided - using embedded GT if available")


if __name__ == '__main__':
    main()