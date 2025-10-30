"""
Fixed HSI Trainer with proper checkpoint saving
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from pathlib import Path

try:
    from .config import config
    from .stage1_lrsr import LRSR
    from .stage2_ebt import EnergyBasedTransformer
    from .data_loader import get_dataloaders
except ImportError:
    from src.config import config
    from src.stage1_lrsr import LRSR
    from src.stage2_ebt import EnergyBasedTransformer
    from src.data_loader import get_dataloaders


class Trainer:
    """HSI Trainer with proper checkpoint saving"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        self.scaler = GradScaler() if config.mixed_precision else None
        self.lrsr = LRSR()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Checkpoint directory: {self.checkpoint_dir}")
        
    def generate_negatives(self, gt):
        """Generate negative samples for contrastive learning"""
        negatives = []
        for i in range(gt.shape[0]):
            neg = gt[i].clone()
            flip_mask = torch.rand_like(neg) < 0.1
            neg[flip_mask] = 1 - neg[flip_mask]
            negatives.append(neg)
        return torch.stack(negatives)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'HSI Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['image'].to(self.device)
                gts = batch['gt'].to(self.device)
                
                # The model expects (B, H, W, C) but loader gives (B, C, H, W)
                images_for_model = images.permute(0, 2, 3, 1)

                # LRSR decomposition
                sparse_components = []
                for img in images_for_model:
                    img_np = img.cpu().numpy()
                    _, S = self.lrsr(img_np)
                    sparse_components.append(torch.from_numpy(S))
                S_batch = torch.stack(sparse_components).to(self.device).float()
                
                # Generate negative samples
                negatives = self.generate_negatives(gts)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                if config.mixed_precision and self.scaler:
                    with autocast():
                        pos_energy = self.model(images_for_model, S_batch, gts)
                        neg_energy = self.model(images_for_model, S_batch, negatives)
                        loss = pos_energy - neg_energy
                    
                    # Check for NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\nWarning: NaN/Inf loss at batch {batch_idx}, skipping...")
                        continue
                    
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pos_energy = self.model(images_for_model, S_batch, gts)
                    neg_energy = self.model(images_for_model, S_batch, negatives)
                    loss = pos_energy - neg_energy
                    
                    # Check for NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\nWarning: NaN/Inf loss at batch {batch_idx}, skipping...")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        self.scheduler.step()
        avg_loss = total_loss / max(valid_batches, 1)
        return avg_loss
    
    def validate(self):
        """Validation phase (placeholder for now)"""
        # TODO: Implement validation
        pass
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        # Save regular checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'hsi_best.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model to {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f'hsi_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved checkpoint to {checkpoint_path}")
    
    def train(self, num_epochs=None):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
        """
        num_epochs = num_epochs or config.num_epochs
        
        print("\n" + "="*60)
        print("STARTING HSI TRAINING")
        print("="*60)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Mixed precision: {config.mixed_precision}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Print epoch summary
            print(f'\nEpoch {epoch}: Train Loss = {train_loss:.4f}')
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        print(f"Best Loss: {best_loss:.4f}")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")
        print("="*60 + "\n")
        
        return best_loss


if __name__ == '__main__':
    print("=" * 60)
    print("HSI TRAINER TEST")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 1. Load Data
    print("\n📊 Loading data...")
    train_loader, val_loader = get_dataloaders(batch_size=2, test_mode=True)

    if train_loader and len(train_loader) > 0:
        print(f"✓ Data loaded: {len(train_loader)} train batches")
        
        # 2. Initialize Model
        print("\n🔧 Initializing model...")
        model = EnergyBasedTransformer()
        print("✓ Model initialized")

        # 3. Initialize Trainer
        print("\n🚀 Creating trainer...")
        hsi_trainer = Trainer(model, train_loader, val_loader, device=device)

        # 4. Run training
        print("\n▶️  Starting training...")
        hsi_trainer.train(num_epochs=50)  # Train for X_No. epochs for testing
        
        print("\n✓ HSI Trainer test finished successfully!")
        
        # Verify checkpoints were saved
        print("\n📁 Checking saved checkpoints...")
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob('hsi_*.pt'))
        
        if checkpoints:
            print(f"✓ Found {len(checkpoints)} checkpoint(s):")
            for cp in checkpoints:
                size_mb = cp.stat().st_size / (1024 * 1024)
                print(f"  • {cp.name} ({size_mb:.2f} MB)")
        else:
            print("⚠️  No checkpoints found!")
            
    else:
        print("\n✗ Could not run HSI trainer test because data loader failed.")
        print("Check your data directory and ensure HSI data exists.")