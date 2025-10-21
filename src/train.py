import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from src.config import config
from src.stage1_lrsr import LRSR
from src.stage2_ebt import EnergyBasedTransformer

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
        # Random flip negatives
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
            
            # Compute LRSR decomposition (cache this in practice)
            sparse_components = []
            for img in images:
                img_np = img.cpu().numpy()
                _, S = self.lrsr(img_np)
                sparse_components.append(torch.from_numpy(S))
            S_batch = torch.stack(sparse_components).to(self.device)
            
            # Generate negative samples
            negatives = self.generate_negatives(gts)
            
            self.optimizer.zero_grad()
            
            if config.mixed_precision:
                with autocast():
                    # Positive energy (should be low)
                    pos_energy = self.model(images, S_batch, gts)
                    
                    # Negative energy (should be high)
                    neg_energy = self.model(images, S_batch, negatives)
                    
                    # Contrastive loss
                    loss = pos_energy - neg_energy
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pos_energy = self.model(images, S_batch, gts)
                neg_energy = self.model(images, S_batch, negatives)
                loss = pos_energy - neg_energy
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        self.scheduler.step()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        # TODO: Implement validation with metrics
        pass
    
    def train(self):
        for epoch in range(config.num_epochs):
            train_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch}: Loss = {train_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        path = f'{config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)    