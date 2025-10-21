import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from pathlib import Path
from src.config import config

class ThermalDataset(Dataset):
    """
    Dataset for thermal infrared anomaly detection
    Handles Landsat thermal bands and other thermal sensors
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """
        Scan the data directory and load thermal dataset information
        TODO: Person 2 - implement scanning logic for thermal datasets
        
        Expected formats:
        - Landsat thermal bands (band 10, 11)
        - FLIR thermal imagery
        - Other thermal sensors from competition links
        """
        samples = []
        
        # Example structure for Landsat
        thermal_dir = self.data_dir / 'thermal' / self.split
        if thermal_dir.exists():
            for thermal_file in thermal_dir.glob('*.tif'):
                # Assume ground truth has same name with _gt suffix
                gt_file = thermal_file.parent / f"{thermal_file.stem}_gt.tif"
                
                if gt_file.exists():
                    samples.append({
                        'thermal_path': str(thermal_file),
                        'gt_path': str(gt_file),
                        'name': thermal_file.stem
                    })
        
        return samples
    
    def _read_thermal_image(self, path):
        """
        Read thermal image and convert to temperature if needed
        Landsat thermal bands store digital numbers that need conversion
        """
        with rasterio.open(path) as src:
            thermal = src.read(1)  # Read first band
            
            # TODO: Person 2 - Add proper temperature conversion for Landsat
            # For Landsat 8/9 thermal bands:
            # 1. Convert DN to radiance using calibration coefficients
            # 2. Convert radiance to brightness temperature
            
            return thermal.astype(np.float32)
    
    def _normalize_thermal(self, thermal):
        """Normalize thermal data to reasonable range"""
        # Remove outliers using percentile clipping
        p_low, p_high = np.percentile(thermal, [2, 98])
        thermal = np.clip(thermal, p_low, p_high)
        
        # Normalize to [0, 1]
        thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-8)
        
        return thermal
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load thermal image and ground truth
        thermal = self._read_thermal_image(sample['thermal_path'])
        gt = self._read_thermal_image(sample['gt_path'])
        
        # Normalize
        thermal = self._normalize_thermal(thermal)
        gt = (gt > 0).astype(np.float32)  # Binarize ground truth
        
        # Apply augmentation if specified
        if self.transform:
            thermal, gt = self.transform(thermal, gt)
        
        # Convert to tensors and add channel dimension
        thermal = torch.from_numpy(thermal).unsqueeze(0)  # (1, H, W)
        gt = torch.from_numpy(gt)  # (H, W)
        
        return {
            'thermal': thermal,
            'gt': gt,
            'name': sample['name']
        }

def get_thermal_dataloaders(batch_size=None):
    """Create thermal data loaders"""
    batch_size = batch_size or config.batch_size
    
    train_dataset = ThermalDataset(config.data_dir, split='train')
    val_dataset = ThermalDataset(config.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader