"""
Enhanced Thermal Training Script - Optimized for Better F1
Includes data augmentation, focal loss, and better hyperparameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.thermal_model import ThermalAnomalyDetector
from src.thermal_data_loader import get_robust_thermal_dataloaders
from src.config import config
from src.metrics import compute_metrics


class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class EnhancedThermalTrainer:
    """Enhanced training with focal loss and better optimization"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Better optimizer settings
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-4,  # Slightly higher for faster convergence
            weight_decay=1e-4,  # Lower weight decay
            betas=(0.9, 0.999)
        )
        
        # Warm-up + cosine schedule
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=2e-4,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # 10% warm-up
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        # Focal Loss instead of BCE (better for imbalanced data)
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Alternative: Dice Loss for better segmentation
        self.use_dice = True
        
        print(f"Enhanced Thermal Trainer initialized on {device}")
        print(f"Optimizer: AdamW with OneCycleLR")
        print(f"Loss: Focal Loss (alpha=0.25, gamma=2.0) + Dice Loss")
        print(f"Learning rate: 2e-4 with warm-up")
        
    def dice_loss(self, pred, target):
        """Dice loss for better segmentation quality"""
        pred = torch.sigmoid(pred)
        smooth = 1.0
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_focal = 0
        total_dice = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Enhanced Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            thermal = batch['thermal'].to(self.device).float()
            gt = batch['gt'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(thermal)
            
            # Combined loss: Focal + Dice
            focal_loss = self.criterion(pred.squeeze(1), gt)
            
            if self.use_dice:
                dice = self.dice_loss(pred.squeeze(1), gt)
                loss = focal_loss + dice
                total_dice += dice.item()
            else:
                loss = focal_loss
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"\nWARNING: NaN loss at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()  # Step every batch for OneCycleLR
            
            total_loss += loss.item()
            total_focal += focal_loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'focal': f'{focal_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_focal = total_focal / max(n_batches, 1)
        avg_dice = total_dice / max(n_batches, 1) if self.use_dice else 0
        
        return avg_loss, avg_focal, avg_dice
    
    def validate(self):
        """Validation with post-processing for better F1"""
        if self.val_loader is None or len(self.val_loader) == 0:
            return None
            
        self.model.eval()
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                thermal = batch['thermal'].to(self.device).float()
                gt = batch['gt'].cpu().numpy()
                
                pred = torch.sigmoid(self.model(thermal))
                pred = pred.squeeze(1).cpu().numpy()
                
                # Check for NaN
                if np.isnan(pred).any():
                    print(f"\nWARNING: NaN in validation predictions")
                    pred = np.nan_to_num(pred, 0.0)
                
                all_preds.append(pred)
                all_gts.append(gt)
        
        all_preds = np.concatenate(all_preds)
        all_gts = np.concatenate(all_gts)
        
        # Find best threshold for F1
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.linspace(0.1, 0.9, 17):
            preds_binary = (all_preds > threshold).astype(float)
            metrics = compute_metrics(preds_binary.flatten(), all_gts.flatten())
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = threshold
        
        # Compute final metrics with best threshold
        preds_binary = (all_preds > best_threshold).astype(float)
        metrics = compute_metrics(preds_binary.flatten(), all_gts.flatten())
        metrics['best_threshold'] = best_threshold
        
        return metrics
    
    def train(self, num_epochs=20):  # More epochs
        best_f1 = 0
        patience = 10
        patience_counter = 0
        
        print(f"\nStarting enhanced thermal training for {num_epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            train_loss, train_focal, train_dice = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            if val_metrics:
                print(f'\nEpoch {epoch}:')
                print(f'  Train Loss: {train_loss:.4f} (Focal: {train_focal:.4f}, Dice: {train_dice:.4f})')
                print(f'  Val F1: {val_metrics["f1"]:.4f} (threshold: {val_metrics["best_threshold"]:.2f})')
                print(f'  Val Precision: {val_metrics["precision"]:.4f}')
                print(f'  Val Recall: {val_metrics["recall"]:.4f}')
                print(f'  Val ROC-AUC: {val_metrics.get("roc_auc", 0.0):.4f}')
                print(f'  Val PR-AUC: {val_metrics.get("pr_auc", 0.0):.4f}')
                print(f'  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}')
                
                # Save best model
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True, metrics=val_metrics)
                    print(f'  ✓ New best F1: {best_f1:.4f}')
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping: No improvement for {patience} epochs")
                    break
            else:
                print(f'\nEpoch {epoch}: Loss={train_loss:.4f}')
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best F1: {best_f1:.4f}")
        print(f"{'='*60}")
    
    def save_checkpoint(self, epoch, is_best=False, metrics=None):
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        filename = 'thermal_best_enhanced.pt' if is_best else f'thermal_epoch_{epoch}_enhanced.pt'
        path = os.path.join(config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            print(f'  Saved enhanced model to {path}')


if __name__ == '__main__':
    print("=" * 60)
    print("THERMAL ANOMALY DETECTION - ENHANCED TRAINING")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading thermal data...")
    train_loader, val_loader = get_robust_thermal_dataloaders(
        batch_size=8,  # Larger batch size
        data_dir='./data',
        target_size=(256, 256)
    )
    
    if not train_loader:
        print("ERROR: No thermal data found!")
        sys.exit(1)
    
    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(val_loader) if val_loader else 0} val batches")
    
    # Initialize model
    print("\nInitializing model...")
    model = ThermalAnomalyDetector(mode='direct')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {total_params/1e6:.2f}M parameters")
    
    # Train with enhanced settings
    trainer = EnhancedThermalTrainer(model, train_loader, val_loader, device=device)
    trainer.train(num_epochs=100)