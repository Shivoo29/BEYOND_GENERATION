import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from src.thermal_model import ThermalAnomalyDetector
from src.config import config

class ThermalTrainer:
    """
    Training pipeline for thermal anomaly detection
    Simpler than hyperspectral because we use standard supervised learning
    """
    
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
        
        # Loss function - weighted BCE due to class imbalance
        pos_weight = torch.tensor([10.0]).to(device)  # Adjust based on dataset
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Thermal Epoch {epoch}')
        for batch in pbar:
            thermal = batch['thermal'].to(self.device)
            gt = batch['gt'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if config.mixed_precision:
                with autocast():
                    pred = self.model(thermal)
                    loss = self.criterion(pred.squeeze(1), gt)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(thermal)
                loss = self.criterion(pred.squeeze(1), gt)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        self.scheduler.step()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validation with metrics computation"""
        self.model.eval()
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                thermal = batch['thermal'].to(self.device)
                gt = batch['gt'].cpu().numpy()
                
                pred = torch.sigmoid(self.model(thermal))
                pred = pred.squeeze(1).cpu().numpy()
                
                all_preds.append(pred)
                all_gts.append(gt)
        
        # Compute metrics using Person 2's metrics module
        from src.metrics import compute_metrics
        metrics = compute_metrics(
            np.concatenate(all_preds).flatten(),
            np.concatenate(all_gts).flatten()
        )
        
        return metrics
    
    def train(self):
        best_f1 = 0
        
        for epoch in range(config.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            print(f'Epoch {epoch}: Loss={train_loss:.4f}, '
                  f'F1={val_metrics["f1"]:.4f}, '
                  f'PR-AUC={val_metrics["pr_auc"]:.4f}')
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_checkpoint(epoch, is_best=True)
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        filename = 'thermal_best.pt' if is_best else f'thermal_epoch_{epoch}.pt'
        path = f'{config.checkpoint_dir}/{filename}'
        torch.save(checkpoint, path)