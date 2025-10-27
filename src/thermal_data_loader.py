import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from pathlib import Path
from src.config import config
import os

class ThermalDataset(Dataset):
    """
    Dataset for thermal infrared anomaly detection.
    Handles 2-band GeoTIFFs where:
    - Band 1: Thermal data (Input)
    - Band 2: Ground Truth Mask (Label)
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()
        print(f"Found {len(self.samples)} thermal scenes for the '{split}' split.")
        
    def _load_samples(self):
        """
        Scan the data directory for the 2-band GeoTIFF files.
        """
        samples = []
        # Point to the directory where the GEE exports were saved.
        # Note: The problem statement does not specify a train/val/test split for this new data.
        # We will create our own split from the downloaded files.
        thermal_dir = self.data_dir / 'raw' / 'HSI_Thermal_Exports'
        
        if not thermal_dir.exists():
            print(f"Warning: Directory not found {thermal_dir}")
            return samples

        all_files = sorted([str(p) for p in thermal_dir.glob('*.tif')])
        
        # Create a deterministic train/val/test split (e.g., 70/15/15)
        np.random.seed(42) # for reproducibility
        np.random.shuffle(all_files)
        
        n_samples = len(all_files)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if self.split == 'train':
            samples = all_files[:n_train]
        elif self.split == 'val':
            samples = all_files[n_train:n_train + n_val]
        else: # test
            samples = all_files[n_train + n_val:]
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath = self.samples[idx]
        
        try:
            with rasterio.open(filepath) as src:
                # Read Band 1 as the thermal image
                thermal_img = src.read(1).astype(np.float32)
                
                # Read Band 2 as the ground truth mask
                gt_mask = src.read(2).astype(np.float32)

        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            # Return dummy data if file is corrupt
            return {
                'thermal': torch.zeros(1, 256, 256),
                'gt': torch.zeros(256, 256),
                'name': os.path.basename(filepath)
            }

        # Normalize thermal image (example: simple min-max)
        p_low, p_high = np.percentile(thermal_img, [2, 98])
        thermal_img = np.clip(thermal_img, p_low, p_high)
        thermal_img = (thermal_img - thermal_img.min()) / (thermal_img.max() - thermal_img.min() + 1e-8)
        
        # Binarize ground truth just in case
        gt_mask = (gt_mask > 0).astype(np.float32)
        
        # TODO: Add tiling logic here if images are large
        # For now, assuming images are of a manageable size or will be resized
        
        # Apply augmentation if specified
        if self.transform:
            # Note: Augmentation needs to be adapted for single-band thermal
            pass

        # Convert to tensors and add channel dimension
        thermal_tensor = torch.from_numpy(thermal_img).unsqueeze(0)  # Shape: [1, H, W]
        gt_tensor = torch.from_numpy(gt_mask)  # Shape: [H, W]
        
        return {
            'thermal': thermal_tensor,
            'gt': gt_tensor,
            'name': os.path.basename(filepath)
        }

def get_thermal_dataloaders(batch_size=None, data_dir=None):
    """Create thermal data loaders for the new 2-band GeoTIFFs."""
    batch_size = batch_size or config.batch_size
    data_dir = data_dir or config.data_dir

    train_dataset = ThermalDataset(data_dir, split='train')
    val_dataset = ThermalDataset(data_dir, split='val')
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Training or validation dataset is empty. Check data directory and splits.")
        return None, None

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
