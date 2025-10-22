#!/usr/bin/env python
"""
Main training script for Energy-Based Transformer
Handles end-to-end training pipeline for the AI Grand Challenge
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import config
from src.stage1_lrsr import LRSR
from src.stage2_ebt import EnergyBasedTransformer
from src.thermal_model import ThermalAnomalyDetector
from src.multimodal_fusion import MultimodalEnergyFunction, MultimodalInference
from src.data_loader import get_dataloaders
from src.thermal_data_loader import get_thermal_dataloaders
from src.train import Trainer
from src.train_thermal import ThermalTrainer
from src.inference import InferencePipeline
from src.metrics import compute_metrics
from src.utils import (
    plot_detection_results,
    create_submission_report,
    compute_file_hash,
    plot_training_history
)

class ComprehensiveTrainer:
    """
    Complete training pipeline for all models
    Manages HSI, thermal, and multimodal fusion training
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.setup_directories()
        
        # Initialize logging
        if args.use_wandb:
            wandb.init(
                project="hyperspectral-ebt",
                config=vars(args),
                name=f"ebt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Training history
        self.history = {
            'loss': [],
            'val_loss': [],
            'f1': [],
            'val_f1': [],
            'pr_auc': [],
            'val_pr_auc': [],
            'lr': []
        }
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            Path(self.args.checkpoint_dir),
            Path(self.args.results_dir),
            Path(self.args.results_dir) / 'visualizations',
            Path(self.args.results_dir) / 'submissions',
            Path(self.args.cache_dir)
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def train_hsi_model(self):
        """Train hyperspectral anomaly detection model"""
        print("=" * 50)
        print("Training Hyperspectral Model (Stage 2 - EBT)")
        print("=" * 50)
        
        # Initialize model
        model = EnergyBasedTransformer().to(self.device)
        
        # Load data
        train_loader, val_loader = get_dataloaders(
            batch_size=self.args.batch_size,
            data_dir=self.args.data_dir
        )
        
        if len(train_loader) == 0:
            print("Warning: No training data found. Skipping HSI training.")
            return None
        
        # Initialize LRSR for Stage 1
        lrsr = LRSR()
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.num_epochs
        )
        
        scaler = GradScaler() if self.args.mixed_precision else None
        
        best_val_pr_auc = 0
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_metrics = {'f1': 0, 'pr_auc': 0}
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.args.num_epochs}')
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                gts = batch['gt'].to(self.device)
                
                # Stage 1: LRSR decomposition (cache this in practice)
                sparse_components = []
                for img in images:
                    img_np = img.cpu().numpy()
                    _, S = lrsr(img_np)
                    sparse_components.append(torch.from_numpy(S).float())
                S_batch = torch.stack(sparse_components).to(self.device)
                
                # Generate negative samples
                negatives = self.generate_negative_samples(gts)
                
                optimizer.zero_grad()
                
                if self.args.mixed_precision and scaler is not None:
                    with autocast():
                        # Positive energy (should be low)
                        pos_energy = model(images, S_batch, gts)
                        # Negative energy (should be high)
                        neg_energy = model(images, S_batch, negatives)
                        # Contrastive loss with margin
                        loss = torch.relu(pos_energy - neg_energy + self.args.margin)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pos_energy = model(images, S_batch, gts)
                    neg_energy = model(images, S_batch, negatives)
                    loss = torch.relu(pos_energy - neg_energy + self.args.margin)
                    
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            # Validation phase
            val_loss, val_metrics = self.validate_model(model, val_loader, lrsr)
            
            # Update history
            self.history['loss'].append(train_loss / len(train_loader))
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_pr_auc'].append(val_metrics['pr_auc'])
            self.history['lr'].append(scheduler.get_last_lr()[0])
            
            # Print epoch summary
            print(f"Epoch {epoch+1}: "
                  f"Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val F1={val_metrics['f1']:.4f}, "
                  f"Val PR-AUC={val_metrics['pr_auc']:.4f}")
            
            # Save best model
            if val_metrics['pr_auc'] > best_val_pr_auc:
                best_val_pr_auc = val_metrics['pr_auc']
                self.save_checkpoint(model, optimizer, epoch, 'hsi_best.pt')
                print(f"New best model saved! PR-AUC: {best_val_pr_auc:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(model, optimizer, epoch, f'hsi_epoch_{epoch+1}.pt')
            
            scheduler.step()
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    'hsi/train_loss': train_loss / len(train_loader),
                    'hsi/val_loss': val_loss,
                    'hsi/val_f1': val_metrics['f1'],
                    'hsi/val_pr_auc': val_metrics['pr_auc'],
                    'hsi/lr': scheduler.get_last_lr()[0]
                })
        
        return model
    
    def train_thermal_model(self):
        """Train thermal anomaly detection model"""
        print("=" * 50)
        print("Training Thermal Model")
        print("=" * 50)
        
        # Initialize model
        model = ThermalAnomalyDetector(mode='direct').to(self.device)
        
        # Load data
        try:
            train_loader, val_loader = get_thermal_dataloaders(
                batch_size=self.args.batch_size
            )
        except:
            print("Warning: No thermal data found. Skipping thermal training.")
            return None
        
        # Train using thermal trainer
        trainer = ThermalTrainer(model, train_loader, val_loader, self.device)
        trainer.train()
        
        return model
    
    def train_multimodal_fusion(self, hsi_model=None, thermal_model=None):
        """Train multimodal fusion model"""
        print("=" * 50)
        print("Training Multimodal Fusion")
        print("=" * 50)
        
        # Initialize fusion model
        fusion_model = MultimodalEnergyFunction().to(self.device)
        
        # Load pre-trained components if available
        if hsi_model is not None:
            fusion_model.hsi_energy_fn.load_state_dict(hsi_model.state_dict())
            print("Loaded pre-trained HSI model")
        
        if thermal_model is not None:
            # Convert thermal model to energy mode
            thermal_energy = ThermalAnomalyDetector(mode='energy').to(self.device)
            thermal_energy.encoder.load_state_dict(thermal_model.encoder.state_dict())
            fusion_model.tir_energy_fn = thermal_energy
            print("Loaded pre-trained thermal model")
        
        # Freeze pre-trained components initially
        if self.args.freeze_pretrained:
            for param in fusion_model.hsi_energy_fn.parameters():
                param.requires_grad = False
            for param in fusion_model.tir_energy_fn.parameters():
                param.requires_grad = False
        
        # TODO: Implement multimodal data loader and training loop
        print("Multimodal fusion training not yet fully implemented")
        
        return fusion_model
    
    def validate_model(self, model, val_loader, lrsr):
        """Validate model and compute metrics"""
        model.eval()
        val_loss = 0
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                gts = batch['gt'].to(self.device)
                
                # LRSR decomposition
                sparse_components = []
                for img in images:
                    img_np = img.cpu().numpy()
                    _, S = lrsr(img_np)
                    sparse_components.append(torch.from_numpy(S).float())
                S_batch = torch.stack(sparse_components).to(self.device)
                
                # Inference with iterative refinement
                inference = InferencePipeline(model, self.device)
                
                for i in range(images.shape[0]):
                    pred = inference.iterative_refinement(
                        images[i:i+1], 
                        S_batch[i:i+1],
                        num_iters=self.args.refinement_iters
                    )
                    all_preds.append(pred)
                    all_gts.append(gts[i].cpu().numpy())
                
                # Compute loss for monitoring
                negatives = self.generate_negative_samples(gts)
                pos_energy = model(images, S_batch, gts)
                neg_energy = model(images, S_batch, negatives)
                loss = torch.relu(pos_energy - neg_energy + self.args.margin)
                val_loss += loss.item()
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_gts = np.array(all_gts)
        
        metrics = compute_metrics(
            all_preds.flatten(),
            all_gts.flatten()
        )
        
        return val_loss / len(val_loader), metrics
    
    def generate_negative_samples(self, gt):
        """Generate negative samples for contrastive learning"""
        negatives = []
        for i in range(gt.shape[0]):
            neg = gt[i].clone()
            
            # Strategy 1: Random flipping
            if torch.rand(1) < 0.5:
                flip_mask = torch.rand_like(neg) < 0.1
                neg[flip_mask] = 1 - neg[flip_mask]
            
            # Strategy 2: Morphological operations
            else:
                if neg.sum() > 0:
                    # Dilate anomalies
                    kernel_size = 3
                    neg = torch.nn.functional.max_pool2d(
                        neg.unsqueeze(0).unsqueeze(0),
                        kernel_size,
                        stride=1,
                        padding=kernel_size//2
                    ).squeeze()
                else:
                    # Add random anomalies
                    neg = (torch.rand_like(neg) < 0.05).float()
            
            negatives.append(neg)
        
        return torch.stack(negatives)
    
    def save_checkpoint(self, model, optimizer, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(self.args)
        }
        
        path = Path(self.args.checkpoint_dir) / filename
        torch.save(checkpoint, path)
        
        # Compute and save model hash
        model_hash = compute_file_hash(str(path))
        hash_file = path.with_suffix('.hash')
        with open(hash_file, 'w') as f:
            f.write(model_hash)
        
        print(f"Saved checkpoint: {filename} (hash: {model_hash[:8]}...)")
    
    def run_complete_pipeline(self):
        """Run complete training pipeline"""
        start_time = time.time()
        
        # Stage 1: Always available (LRSR is parameter-free)
        print("Stage 1: LRSR is parameter-free and ready to use")
        
        # Stage 2: Train HSI model
        hsi_model = None
        if self.args.train_hsi:
            hsi_model = self.train_hsi_model()
        
        # Train thermal model
        thermal_model = None
        if self.args.train_thermal:
            thermal_model = self.train_thermal_model()
        
        # Train multimodal fusion
        fusion_model = None
        if self.args.train_fusion and (hsi_model or thermal_model):
            fusion_model = self.train_multimodal_fusion(hsi_model, thermal_model)
        
        # Save training history
        history_path = Path(self.args.results_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training history
        if len(self.history['loss']) > 0:
            plot_path = Path(self.args.results_dir) / 'training_curves.png'
            plot_training_history(self.history, str(plot_path))
        
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time/3600:.2f} hours")
        
        # Final summary
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Models saved in: {self.args.checkpoint_dir}")
        print(f"Results saved in: {self.args.results_dir}")
        
        if self.args.use_wandb:
            wandb.finish()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Energy-Based Transformer')
    
    # Model selection
    parser.add_argument('--train_hsi', action='store_true', default=True,
                       help='Train hyperspectral model')
    parser.add_argument('--train_thermal', action='store_true', default=False,
                       help='Train thermal model')
    parser.add_argument('--train_fusion', action='store_true', default=False,
                       help='Train multimodal fusion')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--cache_dir', type=str, default='./data/cache',
                       help='Cache directory')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                       help='Weight decay')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Contrastive loss margin')
    
    # Model parameters
    parser.add_argument('--refinement_iters', type=int, default=10,
                       help='Number of refinement iterations')
    parser.add_argument('--freeze_pretrained', action='store_true',
                       help='Freeze pre-trained components in fusion')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./models/checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Print configuration
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Run training
    trainer = ComprehensiveTrainer(args)
    trainer.run_complete_pipeline()