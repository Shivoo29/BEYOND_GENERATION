import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from .config import config
from .stage1_lrsr import LRSR
from .stage2_ebt import EnergyBasedTransformer
from .data_loader import get_dataloaders

class Trainer:
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
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            gts = batch['gt'].to(self.device)
            
            # The model expects (B, H, W, C) but loader gives (B, C, H, W)
            images_for_model = images.permute(0, 2, 3, 1)

            sparse_components = []
            for img in images_for_model:
                img_np = img.cpu().numpy()
                _, S = self.lrsr(img_np)
                sparse_components.append(torch.from_numpy(S))
            S_batch = torch.stack(sparse_components).to(self.device).float()
            
            negatives = self.generate_negatives(gts)
            
            self.optimizer.zero_grad()
            
            if config.mixed_precision and self.scaler:
                with autocast():
                    pos_energy = self.model(images_for_model, S_batch, gts)
                    neg_energy = self.model(images_for_model, S_batch, negatives)
                    loss = pos_energy - neg_energy
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pos_energy = self.model(images_for_model, S_batch, gts)
                neg_energy = self.model(images_for_model, S_batch, negatives)
                loss = pos_energy - neg_energy
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        self.scheduler.step()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        pass
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch}: Loss = {train_loss:.4f}')
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        path = f'{config.checkpoint_dir}/hsi_checkpoint_epoch_{epoch}.pt'
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)
        torch.save(checkpoint, path)
        print(f"Saved HSI checkpoint to {path}")

if __name__ == '__main__':
    print("--- Running a test of the HSI Trainer ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader = get_dataloaders(batch_size=2, test_mode=True)

    if train_loader:
        # 2. Initialize Model
        model = EnergyBasedTransformer()

        # 3. Initialize Trainer
        hsi_trainer = Trainer(model, train_loader, val_loader, device=device)

        # 4. Run for one epoch
        print("Starting a single epoch test...")
        hsi_trainer.train(num_epochs=1)
        print("\n--- HSI Trainer test finished successfully! ---")
    else:
        print("Could not run HSI trainer test because data loader failed.")
