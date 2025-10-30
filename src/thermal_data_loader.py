"""
Improved Thermal Data Loader with Robust Normalization
Prevents NaN/Inf values that cause training instability
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from pathlib import Path
import cv2

class ThermalDatasetRobust(Dataset):
    """
    Robust thermal dataset with improved normalization
    """
    
    def __init__(self, data_dir, split='train', transform=None, has_gt=True, target_size=(256, 256)):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.has_gt = has_gt
        self.target_size = target_size
        self.samples = self._load_samples()
        print(f"Found {len(self.samples)} thermal scenes for '{split}' split")
        
    def _load_samples(self):
        samples = []
        thermal_dir = self.data_dir
        
        if not thermal_dir.exists():
            print(f"Warning: Directory not found {thermal_dir}")
            return samples

        all_files = sorted([str(p) for p in thermal_dir.rglob('*.tif')])
        
        if not all_files:
            print(f"Warning: No .tif files found in {thermal_dir}")
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
    
    def _robust_normalize(self, img):
        """
        Robust normalization that prevents NaN/Inf
        """
        # Remove NaN and Inf
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Clip extreme outliers using percentiles
        p_low, p_high = np.percentile(img[img != 0], [1, 99])  # Ignore zeros
        img_clipped = np.clip(img, p_low, p_high)
        
        # Robust min-max normalization
        img_min = img_clipped.min()
        img_max = img_clipped.max()
        
        if abs(img_max - img_min) < 1e-6:
            # Constant image - return zeros
            return np.zeros_like(img, dtype=np.float32)
        
        normalized = (img_clipped - img_min) / (img_max - img_min)
        
        # Final safety check
        normalized = np.clip(normalized, 0, 1)
        
        return normalized.astype(np.float32)
    
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
            print(f"Error reading {filepath}: {e}")
            return {
                'thermal': torch.zeros(1, *self.target_size, dtype=torch.float32),
                'gt': torch.zeros(self.target_size, dtype=torch.float32),
                'name': 'error'
            }

        # Resize
        if self.target_size:
            thermal_img = cv2.resize(thermal_img, self.target_size, interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Robust normalization
        thermal_img = self._robust_normalize(thermal_img)
        gt_mask = (gt_mask > 0).astype(np.float32)
        
        # Convert to tensors
        thermal_tensor = torch.from_numpy(thermal_img).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_mask)
        
        return {
            'thermal': thermal_tensor,
            'gt': gt_tensor,
            'name': Path(filepath).stem
        }


def get_robust_thermal_dataloaders(batch_size=4, data_dir='./data', target_size=(256, 256)):
    """Create thermal data loaders with robust normalization"""
    local_data_dir = Path(data_dir) / 'raw' / 'thermal'

    train_dataset = ThermalDatasetRobust(local_data_dir, split='train', has_gt=True, target_size=target_size)
    val_dataset = ThermalDatasetRobust(local_data_dir, split='val', has_gt=True, target_size=target_size)
    
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty!")
        return None, None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced for stability
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ Loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader


if __name__ == '__main__':
    print("Testing robust thermal data loader...")
    train, val = get_robust_thermal_dataloaders(batch_size=2, data_dir='./data')
    
    if train and len(train) > 0:
        print("\n✓ Loader test passed!")
        batch = next(iter(train))
        print(f"Thermal shape: {batch['thermal'].shape}")
        print(f"GT shape: {batch['gt'].shape}")
        print(f"Thermal range: [{batch['thermal'].min():.3f}, {batch['thermal'].max():.3f}]")
        print(f"GT unique values: {torch.unique(batch['gt'])}")
    else:
        print("\n✗ Loader test failed!")