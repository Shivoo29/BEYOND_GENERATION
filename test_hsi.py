#!/usr/bin/env python
"""
Test script for HSI anomaly detection model on October 15th mock dataset
Generates competition-ready outputs with proper formatting and metrics
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import scipy.io
import h5py
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.stage1_lrsr import LRSR
from src.stage2_ebt import EnergyBasedTransformer
from src.inference import InferencePipeline
from src.metrics import compute_metrics
from src.utils import (
    compute_file_hash,
    save_geotiff,
    visualize_hyperspectral,
    plot_detection_results,
    create_submission_report
)

class HSITester:
    """Complete testing pipeline for HSI model"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = EnergyBasedTransformer().to(self.device)
        self.load_model(model_path)
        self.model.eval()
        
        # Initialize LRSR and inference pipeline
        self.lrsr = LRSR()
        self.inference = InferencePipeline(self.model, device=self.device)
        
        print("HSI Model loaded successfully!")
    
    def load_model(self, model_path):
        """Load trained model checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    def _load_prisma_he5(self, filepath):
        """
        Load PRISMA L2D .he5 file
        Returns: (data, ground_truth, metadata)
        """
        print(f"Detected PRISMA .he5 file")
        
        try:
            with h5py.File(filepath, 'r') as f:
                swaths = f['HDFEOS']['SWATHS']
                
                # Get VNIR and SWIR cubes from PRS_L2D_HCO
                if 'PRS_L2D_HCO' in swaths:
                    data_fields = swaths['PRS_L2D_HCO']['Data Fields']
                    
                    # Load VNIR (66 bands)
                    vnir = data_fields['VNIR_Cube'][:]  # (1186, 66, 1196)
                    print(f"  VNIR shape: {vnir.shape}")
                    
                    # Load SWIR (173 bands)  
                    swir = data_fields['SWIR_Cube'][:]  # (1186, 173, 1196)
                    print(f"  SWIR shape: {swir.shape}")
                    
                    # Transpose to (H, W, C) format
                    vnir = np.transpose(vnir, (0, 2, 1))  # (1186, 1196, 66)
                    swir = np.transpose(swir, (0, 2, 1))  # (1186, 1196, 173)
                    
                    # Combine VNIR + SWIR
                    data = np.concatenate([vnir, swir], axis=2)  # (1186, 1196, 239)
                    print(f"  Combined HSI shape: {data.shape}")
                    
                    # No ground truth in PRISMA files
                    gt = None
                    
                    metadata = {
                        'filename': os.path.basename(filepath),
                        'n_bands': data.shape[2],
                        'vnir_bands': 66,
                        'swir_bands': 173,
                        'sensor': 'PRISMA'
                    }
                    
                    return data, gt, metadata
                
                else:
                    raise ValueError(f"PRS_L2D_HCO swath not found in {filepath}")
        
        except Exception as e:
            print(f"ERROR loading PRISMA file: {e}")
            raise
    
    def load_hsi_file(self, filepath):
        """
        Robustly load HSI file (.mat or .he5)
        Returns: (data, ground_truth, metadata)
        """
        print(f"Loading HSI file: {filepath}")
        
        # Special handling for PRISMA .he5 files
        if filepath.endswith('.he5') and 'PRS_L2D' in filepath:
            return self._load_prisma_he5(filepath)
        
        common_data_keys = ['data', 'Data', 'array', 'img', 'ImgCube_120', 'X', 'HSI', 'hsi']
        common_gt_keys = ['map', 'Map', 'groundtruth', 'GT', 'gt', 'mask']
        
        try:
            if filepath.endswith('.mat'):
                try:
                    file_data = scipy.io.loadmat(filepath)
                except NotImplementedError:
                    file_data = h5py.File(filepath, 'r')
            elif filepath.endswith('.he5'):
                file_data = h5py.File(filepath, 'r')
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # Find data array
            data = None
            for key in common_data_keys:
                if key in file_data:
                    data = np.array(file_data[key])
                    print(f"Found data with key '{key}', shape: {data.shape}")
                    break
            
            if data is None:
                # Try largest array
                arrays = [(k, v) for k, v in file_data.items() 
                         if isinstance(v, (np.ndarray, h5py.Dataset)) and not k.startswith('__')]
                if arrays:
                    arrays.sort(key=lambda x: x[1].size, reverse=True)
                    data = np.array(arrays[0][1])
                    print(f"Using largest array '{arrays[0][0]}', shape: {data.shape}")
            
            # Find ground truth
            gt = None
            for key in common_gt_keys:
                if key in file_data:
                    gt = np.array(file_data[key])
                    print(f"Found ground truth with key '{key}', shape: {gt.shape}")
                    break
            
            if isinstance(file_data, h5py.File):
                file_data.close()
            
            if data is None:
                raise ValueError(f"Could not find data array in {filepath}")
            
            return data, gt, {'filename': os.path.basename(filepath)}
        
        except Exception as e:
            print(f"ERROR loading file: {e}")
            raise
    
    def preprocess_hsi(self, data):
        """
        Preprocess HSI data to proper format
        Handles various input shapes and converts to (H, W, C)
        """
        print(f"Preprocessing HSI data, input shape: {data.shape}")
        
        # Handle different dimensionalities
        if data.ndim == 2:
            # Assume single band
            data = np.expand_dims(data, -1)
            print(f"Converted 2D to 3D: {data.shape}")
        
        elif data.ndim == 3:
            # Check if bands are in first dimension
            if data.shape[0] < data.shape[2] and data.shape[0] < data.shape[1]:
                data = np.transpose(data, (1, 2, 0))
                print(f"Transposed to (H, W, C): {data.shape}")
        
        # Normalize
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        print(f"Final preprocessed shape: {data.shape}")
        return data.astype(np.float32)
    
    def preprocess_gt(self, gt, target_shape):
        """Preprocess ground truth to match data shape"""
        if gt is None:
            print("No ground truth provided, creating zero mask")
            return np.zeros(target_shape[:2], dtype=np.float32)
        
        # Squeeze extra dimensions
        while gt.ndim > 2:
            gt = gt.squeeze()
        
        # Resize if needed
        if gt.shape != target_shape[:2]:
            print(f"Resizing GT from {gt.shape} to {target_shape[:2]}")
            gt = np.array(Image.fromarray(gt).resize(
                (target_shape[1], target_shape[0]), 
                Image.NEAREST
            ))
        
        # Binarize
        gt = (gt > 0).astype(np.float32)
        return gt
    
    def run_inference(self, hsi_data, num_iters=10):
        """
        Run full inference pipeline on HSI data
        
        Args:
            hsi_data: numpy array (H, W, C)
            num_iters: number of refinement iterations
            
        Returns:
            anomaly_map: numpy array (H, W)
            processing_time: float
        """
        start_time = time.time()
        
        print("Step 1: LRSR decomposition...")
        L, S = self.lrsr.decompose(hsi_data)
        
        print("Step 2: Converting to tensors...")
        # Convert to torch tensors
        X_tensor = torch.from_numpy(hsi_data).unsqueeze(0).to(self.device).float()
        S_tensor = torch.from_numpy(S).unsqueeze(0).to(self.device).float()
        
        print(f"Step 3: Energy-based refinement ({num_iters} iterations)...")
        anomaly_map = self.inference.iterative_refinement(
            X_tensor, 
            S_tensor, 
            num_iters=num_iters
        )
        
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
            if filepath.endswith('.mat') or filepath.endswith('.he5'):
                # Load from .mat/.he5 file
                common_gt_keys = ['map', 'Map', 'groundtruth', 'GT', 'gt', 'mask', 'label']
                
                if filepath.endswith('.mat'):
                    try:
                        file_data = scipy.io.loadmat(filepath)
                    except NotImplementedError:
                        file_data = h5py.File(filepath, 'r')
                else:
                    file_data = h5py.File(filepath, 'r')
                
                gt = None
                for key in common_gt_keys:
                    if key in file_data:
                        gt = np.array(file_data[key])
                        print(f"Loaded GT from key '{key}', shape: {gt.shape}")
                        break
                
                if isinstance(file_data, h5py.File):
                    file_data.close()
                
                if gt is None:
                    raise ValueError(f"No ground truth found in {filepath}")
                
            elif filepath.endswith('.npy'):
                gt = np.load(filepath)
                print(f"Loaded GT from numpy, shape: {gt.shape}")
                
            elif filepath.endswith('.png') or filepath.endswith('.jpg'):
                gt = np.array(Image.open(filepath))
                print(f"Loaded GT from image, shape: {gt.shape}")
            
            else:
                raise ValueError(f"Unsupported GT format: {filepath}")
            
            # Preprocess
            gt = self.preprocess_gt(gt, data_shape)
            return gt
            
        except Exception as e:
            print(f"WARNING: Could not load ground truth from {filepath}: {e}")
            return None
    
    def test_single_file(self, filepath, output_dir, num_iters=10, gt_filepath=None):
        """
        Test model on a single HSI file
        
        Args:
            filepath: path to HSI file
            output_dir: directory to save results
            num_iters: number of refinement iterations
            gt_filepath: optional separate ground truth file path
            
        Returns:
            results dictionary with metrics and paths
        """
        print("\n" + "="*60)
        print(f"Testing: {filepath}")
        print("="*60)
        
        # Load data
        hsi_data, gt_embedded, metadata = self.load_hsi_file(filepath)
        
        # Preprocess
        hsi_data = self.preprocess_hsi(hsi_data)
        
        # Load ground truth - prioritize separate GT file if provided
        if gt_filepath and os.path.exists(gt_filepath):
            print(f"Loading separate ground truth from: {gt_filepath}")
            gt = self.load_ground_truth(gt_filepath, hsi_data.shape)
        else:
            print("Using embedded ground truth (if available)")
            gt = self.preprocess_gt(gt_embedded, hsi_data.shape)
        
        # Run inference
        anomaly_map, processing_time = self.run_inference(hsi_data, num_iters)
        
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
            'hsi': hsi_data
        }
        
        # 1. Save anomaly map as GeoTIFF
        geotiff_path = output_dir / f"{file_stem}_anomaly.tif"
        save_geotiff(anomaly_map, str(geotiff_path))
        results['geotiff_path'] = str(geotiff_path)
        print(f"\n✓ Saved GeoTIFF: {geotiff_path}")
        
        # 2. Save anomaly map as PNG
        png_path = output_dir / f"{file_stem}_anomaly.png"
        anomaly_uint8 = (anomaly_map * 255).astype(np.uint8)
        Image.fromarray(anomaly_uint8).save(png_path)
        results['png_path'] = str(png_path)
        print(f"✓ Saved PNG: {png_path}")
        
        # 3. Save visualization
        if gt is not None:
            vis_path = output_dir / f"{file_stem}_visualization.png"
            plot_detection_results(
                hsi_data, 
                gt, 
                anomaly_map > 0.5,
                scores=anomaly_map,
                save_path=str(vis_path)
            )
            results['visualization_path'] = str(vis_path)
            print(f"✓ Saved visualization: {vis_path}")
        
        return results
    
    def test_directory(self, data_dir, output_dir, num_iters=10, gt_dir=None):
        """
        Test model on all HSI files in a directory
        
        Args:
            data_dir: directory containing HSI files
            output_dir: directory to save results
            num_iters: number of refinement iterations
            gt_dir: optional directory containing separate ground truth files
            
        Returns:
            summary_df: pandas DataFrame with results for all files
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all HSI files
        hsi_files = list(data_dir.rglob('*.mat')) + list(data_dir.rglob('*.he5'))
        
        if not hsi_files:
            print(f"No HSI files found in {data_dir}")
            return None
        
        print(f"\nFound {len(hsi_files)} HSI files to process")
        
        all_results = []
        
        for filepath in hsi_files:
            try:
                # Try to find corresponding GT file
                gt_filepath = None
                if gt_dir:
                    gt_dir_path = Path(gt_dir)
                    # Try multiple naming conventions for GT files
                    stem = filepath.stem
                    possible_gt_names = [
                        f"{stem}_gt.mat",
                        f"{stem}_GT.mat",
                        f"{stem}_groundtruth.mat",
                        f"{stem}_map.mat",
                        f"Ground_truth_{stem}.mat",
                        f"{stem}.mat",  # Same name in different directory
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
                    num_iters,
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
                continue
        
        # Create summary DataFrame
        if all_results:
            summary_df = pd.DataFrame(all_results)
            
            # Save summary
            summary_path = output_dir / 'test_summary.xlsx'
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
        checkpoint_files = list(Path('models/checkpoints').glob('hsi_best.pt'))
        if checkpoint_files:
            model_hash = compute_file_hash(str(checkpoint_files[0]))
            print(f"Model hash: {model_hash}")
        
        # 2. Create Excel report
        excel_path = output_dir / f"{team_id}_{timestamp}_HSI_report.xlsx"
        
        with pd.ExcelWriter(excel_path) as writer:
            # Metrics sheet
            if 'metrics' in results_dict and results_dict['metrics']:
                metrics_df = pd.DataFrame([results_dict['metrics']])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Configuration sheet
            config_data = {
                'Team ID': [team_id],
                'Timestamp': [timestamp],
                'Model': ['Energy-Based Transformer'],
                'Dataset': [results_dict.get('filename', 'Unknown')],
                'Processing Time (s)': [results_dict.get('processing_time', 'N/A')],
                'Hardware': [self._get_hardware_info()]
            }
            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name='Configuration', index=False)
            
            # Model hash sheet
            hash_data = {
                'Model File': ['hsi_best.pt'],
                'SHA-256 Hash': [model_hash]
            }
            hash_df = pd.DataFrame(hash_data)
            hash_df.to_excel(writer, sheet_name='Model Hash', index=False)
        
        print(f"✓ Saved Excel report: {excel_path}")
        
        # 3. Create hash file
        hash_path = output_dir / f"{team_id}_{timestamp}_model_hash.txt"
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
    parser = argparse.ArgumentParser(description='Test HSI anomaly detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (e.g., models/checkpoints/hsi_best.pt)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (file or directory)')
    parser.add_argument('--gt', type=str, default=None,
                       help='Path to ground truth file (for single file) or directory (for batch)')
    parser.add_argument('--output', type=str, default='./results/hsi_test',
                       help='Output directory for results')
    parser.add_argument('--iters', type=int, default=10,
                       help='Number of refinement iterations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--team-id', type=str, default='TEAM_EBT',
                       help='Team identifier for submission')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = HSITester(args.model, device=args.device)
    
    # Test data
    data_path = Path(args.data)
    
    if data_path.is_file():
        # Single file
        results = tester.test_single_file(
            str(data_path),
            args.output,
            num_iters=args.iters,
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
            num_iters=args.iters,
            gt_dir=args.gt
        )
        
        if summary_df is not None and len(summary_df) > 0:
            # Create submission for first file (or you can modify to create for all)
            print("\nNote: Creating submission for first processed file")
    
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