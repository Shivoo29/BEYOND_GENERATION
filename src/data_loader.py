import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from src.config import config

class HyperspectralDataset(Dataset):
    """Dataset for hyperspectral anomaly detection"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load all dataset samples - IMPLEMENT THIS"""
        # TODO: Person 2 - scan data_dir and create list of (image_path, gt_path) tuples
        samples = []
        # Example structure:
        # samples.append({
        #     'image_path': 'path/to/hsi.npy',
        #     'gt_path': 'path/to/groundtruth.npy',
        #     'name': 'abu_airport'
        # })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load hyperspectral image and ground truth
        image = np.load(sample['image_path'])  # (H, W, B)
        gt = np.load(sample['gt_path'])  # (H, W)
        
        # Apply preprocessing/augmentation
        if self.transform:
            image, gt = self.transform(image, gt)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        gt = torch.from_numpy(gt).float()
        
        return {
            'image': image,
            'gt': gt,
            'name': sample['name']
        }

def get_dataloaders(batch_size=None):
    """Create train and validation dataloaders"""
    batch_size = batch_size or config.batch_size
    
    train_dataset = HyperspectralDataset(config.data_dir, split='train')
    val_dataset = HyperspectralDataset(config.data_dir, split='val')
    
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