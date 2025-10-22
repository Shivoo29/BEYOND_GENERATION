"""
Updated data loader for your specific dataset structure
Handles the abu, salinas, hydice, and pavia datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import scipy.io
import h5py
from typing import Dict, List, Tuple, Optional

class HyperspectralDataset(Dataset):
    """Dataset loader for your specific file structure"""
    
    DATASET_CONFIGS = {
        'abu_airport': {
            'files': ['abu-airport-1.mat', 'abu-airport-2.mat', 'abu-airport-3.mat', 
                     'abu-airport-4.mat', 'Airport.mat'],
            'data_key': ['data', 'Data', 'array'],
            'gt_key': ['map', 'Map', 'groundtruth', 'GT']
        },
        'abu_beach': {
            'files': ['abu-beach-1.mat', 'abu-beach-2.mat', 'abu-beach-3.mat', 
                     'abu-beach-4.mat', 'Beach.mat'],
            'data_key': ['data', 'Data', 'array'],
            'gt_key': ['map', 'Map', 'groundtruth', 'GT']
        },
        'abu_urban': {
            'files': ['abu-urban-1.mat', 'abu-urban-2.mat', 'abu-urban-3.mat',
                     'abu-urban-4.mat', 'abu-urban-5.mat'],
            'data_key': ['data', 'Data', 'array'],
            'gt_key': ['map', 'Map', 'groundtruth', 'GT']
        },
        'hydice_urban': {
            'files': ['HYDICE_urban.mat'],
            'data_key': ['data', 'Data', 'array', 'img'],
            'gt_key': ['map', 'Map', 'groundtruth', 'GT', 'gt']
        },
        'paviaC': {
            'files': ['paviac_data.emf'],  # Note: EMF format needs special handling
            'data_key': ['data'],
            'gt_key': ['gt']
        },
        'salinas': {
            'files': ['Salinas_120_sj25_gt.mat', 'Salinas_120_120.hdr'],
            'data_key': ['ImgCube_120', 'data', 'Data'],
            'gt_key': ['GT', 'gt', 'groundtruth']
        },
        'san_diego': {
            'files': ['SanDiego.mat', 'SandiegoData.mat'],
            'data_key': ['data', 'Data', 'array'],
            'gt_key': ['map', 'Map', 'groundtruth', 'GT']
        }
    }
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform = None,
                 preprocessor = None,
                 tile_size: int = 256,
                 stride: int = 128,
                 test_mode: bool = False):
        """
        Args:
            data_dir: Root directory (should contain raw/hsi/)
            split: 'train', 'val', or 'test'
            transform: Augmentation pipeline
            preprocessor: Preprocessing pipeline
            tile_size: Size of tiles to extract
            stride: Stride for tile extraction
            test_mode: If True, use smaller subset for testing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.preprocessor = preprocessor
        self.tile_size = tile_size
        self.stride = stride if split == 'train' else tile_size
        self.test_mode = test_mode
        
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_mat_file_robust(self, filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Robustly load .mat files, trying different keys and methods
        Returns: (data, ground_truth) or (None, None) if failed
        """
        try:
            # Try scipy.io first
            mat_data = scipy.io.loadmat(filepath)
            
            # Common data keys to try
            data_keys = ['data', 'Data', 'array', 'img', 'ImgCube_120', 'X', 'HSI', 'hsi']
            gt_keys = ['map', 'Map', 'groundtruth', 'GT', 'gt', 'mask']
            
            data = None
            gt = None
            
            # Find data
            for key in data_keys:
                if key in mat_data and isinstance(mat_data[key], np.ndarray):
                    if mat_data[key].ndim >= 2:
                        data = mat_data[key]
                        break
            
            # Find ground truth
            for key in gt_keys:
                if key in mat_data and isinstance(mat_data[key], np.ndarray):
                    gt = mat_data[key]
                    break
            
            # If not found, try to find the largest arrays
            if data is None:
                arrays = [(k, v) for k, v in mat_data.items() 
                         if isinstance(v, np.ndarray) and not k.startswith('__')]
                if arrays:
                    # Sort by size and take the largest as data
                    arrays.sort(key=lambda x: x[1].size, reverse=True)
                    data = arrays[0][1]
                    if len(arrays) > 1:
                        gt = arrays[1][1]
            
            return data, gt
            
        except NotImplementedError:
            # Try h5py for v7.3 mat files
            try:
                with h5py.File(filepath, 'r') as f:
                    # Similar key search for HDF5
                    data = None
                    gt = None
                    
                    for key in ['data', 'Data', 'array', 'img']:
                        if key in f:
                            data = np.array(f[key])
                            break
                    
                    for key in ['map', 'Map', 'groundtruth', 'GT', 'gt']:
                        if key in f:
                            gt = np.array(f[key])
                            break
                    
                    return data, gt
            except:
                pass
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
        
        return None, None
    
    def _scan_available_data(self) -> Dict[str, List[Dict]]:
        """Scan the data directory for available datasets"""
        available = {}
        hsi_dir = self.data_dir / 'raw' / 'hsi'
        
        if not hsi_dir.exists():
            print(f"Warning: HSI directory {hsi_dir} not found")
            return available
        
        # Scan each dataset folder
        for dataset_name in self.DATASET_CONFIGS.keys():
            dataset_path = hsi_dir / dataset_name
            
            if dataset_path.exists():
                files_found = []
                
                # Look for .mat files
                for mat_file in dataset_path.glob('*.mat'):
                    data, gt = self._load_mat_file_robust(str(mat_file))
                    
                    if data is not None:
                        files_found.append({
                            'file': str(mat_file),
                            'data': data,
                            'gt': gt,
                            'name': mat_file.stem
                        })
                
                if files_found:
                    available[dataset_name] = files_found
                    print(f"Found {len(files_found)} files in {dataset_name}")
        
        return available
    
    def _create_tiles(self, data: np.ndarray, gt: Optional[np.ndarray]) -> List[Dict]:
        """Create tiles from full image"""
        tiles = []
        
        # Ensure correct dimension order (H, W, C)
        if data.ndim == 3:
            if data.shape[0] < data.shape[2]:  # (C, H, W) format
                data = np.transpose(data, (1, 2, 0))
        elif data.ndim == 2:
            data = np.expand_dims(data, -1)
        
        H, W = data.shape[:2]
        
        # Create synthetic GT if none provided
        if gt is None:
            print("Warning: No ground truth found, creating synthetic anomalies")
            gt = (np.random.random((H, W)) > 0.98).astype(np.float32)
        
        # Ensure GT is 2D
        if gt.ndim == 3:
            gt = gt.squeeze()
        
        # Binarize GT
        gt = (gt > 0).astype(np.float32)
        
        # If image is smaller than tile size, pad it
        if H < self.tile_size or W < self.tile_size:
            pad_h = max(0, self.tile_size - H)
            pad_w = max(0, self.tile_size - W)
            data = np.pad(data, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode='constant')
            H, W = data.shape[:2]
        
        # Extract tiles
        for h in range(0, H - self.tile_size + 1, self.stride):
            for w in range(0, W - self.tile_size + 1, self.stride):
                tile_data = data[h:h+self.tile_size, w:w+self.tile_size]
                tile_gt = gt[h:h+self.tile_size, w:w+self.tile_size]
                
                # For training, keep tiles with anomalies
                # For val/test, keep all tiles
                if self.split != 'train' or tile_gt.sum() > 0:
                    tiles.append({'data': tile_data, 'gt': tile_gt})
        
        return tiles
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples"""
        samples = []
        available_data = self._scan_available_data()
        
        if not available_data:
            print("Warning: No data found. Check your data directory structure.")
            print(f"Expected path: {self.data_dir / 'raw' / 'hsi'}")
            return samples
        
        # Process each dataset
        for dataset_name, file_list in available_data.items():
            for file_info in file_list:
                data = file_info['data']
                gt = file_info['gt']
                
                # Normalize data
                if self.preprocessor:
                    from src.preprocessing import HyperspectralPreprocessor
                    preprocessor = HyperspectralPreprocessor()
                    data = preprocessor(data)
                else:
                    # Simple normalization
                    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                
                # Create tiles
                tiles = self._create_tiles(data, gt)
                
                # Add metadata
                for tile in tiles:
                    tile['dataset'] = dataset_name
                    tile['source_file'] = file_info['name']
                    samples.append(tile)
        
        # Split into train/val/test
        np.random.seed(42)
        np.random.shuffle(samples)
        
        n_samples = len(samples)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if self.split == 'train':
            samples = samples[:n_train]
        elif self.split == 'val':
            samples = samples[n_train:n_train + n_val]
        else:  # test
            samples = samples[n_train + n_val:]
        
        # For test mode, use only first 10 samples
        if self.test_mode:
            samples = samples[:10]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        data = sample['data'].copy()
        gt = sample['gt'].copy()
        
        # Apply augmentation
        if self.transform and self.split == 'train':
            data, gt = self.transform(data, gt)
        
        # Convert to tensors
        data_tensor = torch.from_numpy(data).float()
        gt_tensor = torch.from_numpy(gt).float()
        
        return {
            'image': data_tensor,
            'gt': gt_tensor,
            'name': f"{sample['dataset']}_{sample['source_file']}"
        }


def get_dataloaders(batch_size: int = 8,
                   data_dir: str = './data',
                   num_workers: int = 4,
                   test_mode: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for your specific data structure
    
    Args:
        batch_size: Batch size for training
        data_dir: Root data directory
        num_workers: Number of parallel workers
        test_mode: If True, use small subset for testing
    
    Returns:
        train_loader, val_loader
    """
    # Import preprocessing
    try:
        from src.preprocessing import HyperspectralPreprocessor, DataAugmentation
        preprocessor = HyperspectralPreprocessor(normalize_method='minmax')
        augmentor = DataAugmentation(flip_prob=0.5, rotate_prob=0.3)
    except:
        preprocessor = None
        augmentor = None
        print("Warning: Preprocessing module not found, using basic normalization")
    
    # Create datasets
    train_dataset = HyperspectralDataset(
        data_dir=data_dir,
        split='train',
        transform=augmentor,
        preprocessor=preprocessor,
        tile_size=256,
        stride=128,
        test_mode=test_mode
    )
    
    val_dataset = HyperspectralDataset(
        data_dir=data_dir,
        split='val',
        transform=None,
        preprocessor=preprocessor,
        tile_size=256,
        stride=256,
        test_mode=test_mode
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader with your data structure...")
    train_loader, val_loader = get_dataloaders(batch_size=2, test_mode=True)
    
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print(f"Successfully loaded data!")
        print(f"Image shape: {batch['image'].shape}")
        print(f"GT shape: {batch['gt'].shape}")
        print(f"Sample names: {batch['name']}")
    else:
        print("No data loaded. Check your data directory structure.")