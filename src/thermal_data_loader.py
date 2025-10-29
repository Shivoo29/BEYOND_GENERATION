import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from pathlib import Path
from .config import config
import os
import kagglehub
import cv2

class ThermalDataset(Dataset):
    """
    Dataset for thermal infrared anomaly detection.
    Handles GeoTIFFs, resizing them to a uniform target size.
    """
    
    def __init__(self, data_dir, split='train', transform=None, has_gt=True, target_size=(256, 256)):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.has_gt = has_gt
        self.target_size = target_size
        self.samples = self._load_samples()
        print(f"Found {len(self.samples)} thermal scenes for the '{split}' split from '{self.data_dir}'.")
        
    def _load_samples(self):
        """
        Scan the data directory recursively for GeoTIFF files.
        """
        samples = []
        thermal_dir = self.data_dir
        
        if not thermal_dir.exists():
            print(f"Warning: Directory not found {thermal_dir}")
            return samples

        all_files = sorted([str(p) for p in thermal_dir.rglob('*.tif')])
        
        if not all_files:
            print(f"Warning: No .tif files found in {thermal_dir} or its subdirectories.")
            return samples
            
        np.random.seed(42)
        np.random.shuffle(all_files)
        
        n_samples = len(all_files)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if self.split == 'train':
            samples = all_files[:n_train]
        elif self.split == 'val':
            samples = all_files[n_train:n_train + n_val]
        else:
            samples = all_files[n_train + n_val:]
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath = self.samples[idx]
        
        try:
            with rasterio.open(filepath) as src:
                if self.has_gt and src.count >= 2:
                    thermal_img = src.read(1).astype(np.float32)
                    gt_mask = src.read(2).astype(np.float32)
                else:
                    thermal_img = src.read(1).astype(np.float32)
                    gt_mask = np.zeros_like(thermal_img, dtype=np.float32)

        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return {
                'thermal': torch.zeros(1, *self.target_size),
                'gt': torch.zeros(self.target_size),
                'name': os.path.basename(filepath)
            }

        # --- Resize to uniform dimensions ---
        if self.target_size:
            thermal_img = cv2.resize(thermal_img, self.target_size, interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Normalize thermal image
        p_low, p_high = np.percentile(thermal_img, [2, 98])
        thermal_img = np.clip(thermal_img, p_low, p_high)
        thermal_img = (thermal_img - thermal_img.min()) / (thermal_img.max() - thermal_img.min() + 1e-8)
        
        gt_mask = (gt_mask > 0).astype(np.float32)
        
        if self.transform:
            pass

        thermal_tensor = torch.from_numpy(thermal_img).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_mask)
        
        return {
            'thermal': thermal_tensor,
            'gt': gt_tensor,
            'name': os.path.basename(filepath)
        }

def get_local_thermal_dataloaders(batch_size=None, data_dir=None, target_size=(256, 256)):
    """Create thermal data loaders for local GeoTIFFs."""
    batch_size = batch_size or config.batch_size
    local_data_dir = Path(data_dir or config.data_dir) / 'raw' / 'thermal'

    train_dataset = ThermalDataset(local_data_dir, split='train', has_gt=True, target_size=target_size)
    val_dataset = ThermalDataset(local_data_dir, split='val', has_gt=True, target_size=target_size)
    
    if len(train_dataset) == 0:
        print("Training dataset for local thermal data is empty. Check data directory.")
        return None, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    return train_loader, val_loader

def get_kaggle_thermal_dataloaders(batch_size=None, target_size=(256, 256)):
    """Creates thermal data loaders for the FLAME 3 Kaggle dataset."""
    print("Setting up Kaggle thermal dataset: 'brycehopkins/flame-3-nadir-thermal-plot-subset'")
    
    try:
        dataset_path = kagglehub.load_dataset("brycehopkins/flame-3-nadir-thermal-plot-subset")
        print(f"Kaggle dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Failed to download Kaggle dataset. Please ensure you have run 'kaggle login'. Error: {e}")
        return None, None

    batch_size = batch_size or config.batch_size

    train_dataset = ThermalDataset(dataset_path, split='train', has_gt=False, target_size=target_size)
    val_dataset = ThermalDataset(dataset_path, split='val', has_gt=False, target_size=target_size)

    if len(train_dataset) == 0:
        print("Kaggle thermal dataset is empty after splitting. Check downloaded files.")
        return None, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    return train_loader, val_loader

if __name__ == '__main__':
    print("--- Testing Local Thermal Data Loader ---")
    local_train, local_val = get_local_thermal_dataloaders(batch_size=2, data_dir='./data')
    if local_train and len(local_train) > 0:
        print("Local thermal loader OK.")
        batch = next(iter(local_train))
        print(f"Image batch shape: {batch['thermal'].shape}")
    else:
        print("Could not initialize local thermal loader. Check the contents of ./data/raw/thermal/")

    print("\n--- Testing Kaggle Thermal Data Loader ---")
    print("This will attempt to download the dataset from Kaggle.")
    kaggle_train, kaggle_val = get_kaggle_thermal_dataloaders(batch_size=2)
    if kaggle_train and len(kaggle_train) > 0:
        print("Kaggle thermal loader OK.")
        batch = next(iter(kaggle_train))
        print(f"Image batch shape: {batch['thermal'].shape}")
    else:
        print("Could not initialize Kaggle thermal loader. Please check your Kaggle API setup.")
