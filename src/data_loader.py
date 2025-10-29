"""
Updated data loader for HSI datasets.
Handles .mat and .he5 files, automatically reshapes 2D data to 3D,
and pads all samples to have a uniform number of channels.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import scipy.io
import h5py
from PIL import Image
from typing import Dict, List, Tuple, Optional

class HyperspectralDataset(Dataset):
    """Dataset loader for your specific file structure"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform = None,
                 preprocessor = None,
                 tile_size: int = 256,
                 stride: int = 128,
                 test_mode: bool = False):

        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.preprocessor = preprocessor
        self.tile_size = tile_size
        self.stride = stride if split == 'train' else tile_size
        self.test_mode = test_mode
        
        # Load samples and determine max bands for padding
        self.samples = self._load_samples()
        self.max_bands = self._get_max_bands()
        if self.max_bands > 0:
            print(f"Found datasets with varying bands. All samples will be padded to {self.max_bands} bands.")

        print(f"Loaded {len(self.samples)} HSI samples for {self.split} split")
    
    def _get_max_bands(self) -> int:
        """Iterate through all samples to find the max number of bands."""
        max_bands = 0
        if not self.samples:
            return 0
        for sample in self.samples:
            bands = sample['data'].shape[2]
            if bands > max_bands:
                max_bands = bands
        return max_bands

    def _load_file_robust(self, filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int]]:
        """
        Robustly load .mat or .he5 files, trying different keys.
        Returns: (data, ground_truth, height, width) or (None, None, None, None)
        """
        data, gt, h, w = None, None, None, None
        
        common_data_keys = ['data', 'Data', 'array', 'img', 'ImgCube_120', 'X', 'HSI', 'hsi']
        common_gt_keys = ['map', 'Map', 'groundtruth', 'GT', 'gt', 'mask']
        common_h_keys = ['h', 'height', 'lines']
        common_w_keys = ['w', 'width', 'samples']

        try:
            if filepath.endswith('.mat'):
                try:
                    file_data = scipy.io.loadmat(filepath)
                except NotImplementedError:
                    file_data = h5py.File(filepath, 'r')
            elif filepath.endswith('.he5'):
                file_data = h5py.File(filepath, 'r')
            else:
                return None, None, None, None

            for key in common_data_keys:
                if key in file_data and isinstance(file_data[key], (np.ndarray, h5py.Dataset)):
                    data = np.array(file_data[key])
                    break
            
            for key in common_gt_keys:
                if key in file_data and isinstance(file_data[key], (np.ndarray, h5py.Dataset)):
                    gt = np.array(file_data[key])
                    break

            for key in common_h_keys:
                if key in file_data:
                    h = int(np.array(file_data[key]).squeeze())
                    break
            
            for key in common_w_keys:
                if key in file_data:
                    w = int(np.array(file_data[key]).squeeze())
                    break

            if data is None:
                arrays = [(k, v) for k, v in file_data.items() 
                         if isinstance(v, (np.ndarray, h5py.Dataset)) and not k.startswith('__')]
                if arrays:
                    arrays.sort(key=lambda x: x[1].size, reverse=True)
                    data = np.array(arrays[0][1])
                    if len(arrays) > 1 and gt is None:
                        gt = np.array(arrays[1][1])

            if isinstance(file_data, h5py.File):
                file_data.close()

            return data, gt, h, w

        except Exception as e:
            print(f"ERROR: Could not load file {filepath}: {e}")
            return None, None, None, None

    def _scan_available_data(self) -> List[Dict]:
        """Scan the data directory for available datasets."""
        all_files = []
        hsi_dir = self.data_dir / 'raw' / 'hsi'
        mock_dir = self.data_dir.parent / 'mock_dataset_15_oct'

        if hsi_dir.exists():
            for ext in ['*.mat', '*.he5']:
                all_files.extend(hsi_dir.rglob(ext))
        else:
            print(f"Warning: HSI directory {hsi_dir} not found")

        if mock_dir.exists():
            for ext in ['*.mat', '*.he5']:
                all_files.extend(mock_dir.rglob(ext))
            print(f"Found mock dataset directory: {mock_dir}")
        
        files_found = []
        for f in all_files:
            data, gt, h, w = self._load_file_robust(str(f))
            if data is not None:
                files_found.append({
                    'file': str(f),
                    'data': data,
                    'gt': gt,
                    'height': h,
                    'width': w,
                    'name': f.stem
                })
        print(f"Scanned and found {len(files_found)} valid HSI data files.")
        return files_found
    
    def _create_tiles(self, data: np.ndarray, gt: Optional[np.ndarray], h: Optional[int], w: Optional[int], filename: str) -> List[Dict]:
        """Create tiles from a full image, handling dimensionality."""
        if data.ndim == 2 and h is not None and w is not None:
            if data.shape[0] == h * w:
                print(f"Info ({filename}): Reshaping 2D data ({data.shape}) to 3D ({h}, {w}, -1) using metadata.")
                data = data.reshape(h, w, -1)
            else:
                print(f"Warning ({filename}): H*W from metadata ({h*w}) does not match data size ({data.shape[0]}). Cannot reshape.")

        if data.ndim == 3:
            if data.shape[0] < data.shape[2] and data.shape[0] < data.shape[1]:
                data = np.transpose(data, (1, 2, 0))
        elif data.ndim == 2:
            print(f"Warning ({filename}): Data is 2D and could not be reshaped. Assuming single channel image.")
            data = np.expand_dims(data, -1)
        
        H, W = data.shape[:2]
        
        if gt is None:
            gt = np.zeros((H, W), dtype=np.float32)
        
        if gt.ndim == 3:
            gt = gt.squeeze()
        if gt.shape != (H, W):
            print(f"Warning ({filename}): GT shape {gt.shape} mismatch with data shape {(H,W)}. Resizing GT.")
            gt = np.array(Image.fromarray(gt).resize((W, H), Image.NEAREST))

        gt = (gt > 0).astype(np.float32)
        
        tiles = []
        if H < self.tile_size or W < self.tile_size:
            pad_h = max(0, self.tile_size - H)
            pad_w = max(0, self.tile_size - W)
            data = np.pad(data, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode='constant')
            H, W = data.shape[:2]
        
        for h_start in range(0, H - self.tile_size + 1, self.stride):
            for w_start in range(0, W - self.tile_size + 1, self.stride):
                tile_data = data[h_start:h_start+self.tile_size, w_start:w_start+self.tile_size]
                tile_gt = gt[h_start:h_start+self.tile_size, w_start:w_start+self.tile_size]
                
                if self.split != 'train' or tile_gt.sum() > 0 or np.random.random() < 0.1:
                    tiles.append({'data': tile_data, 'gt': tile_gt})
        return tiles
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from all found data files."""
        samples = []
        available_data = self._scan_available_data()
        
        if not available_data:
            print("CRITICAL: No HSI data found. Check your data directory structure.")
            return samples
        
        for file_info in available_data:
            data, gt = file_info['data'], file_info['gt']
            h, w = file_info['height'], file_info['width']
            
            # Handle dimensionality before preprocessing
            if data.ndim == 2 and h is not None and w is not None and data.shape[0] == h * w:
                data = data.reshape(h, w, -1)
            elif data.ndim == 3 and data.shape[0] < data.shape[2] and data.shape[0] < data.shape[1]:
                data = np.transpose(data, (1, 2, 0))
            elif data.ndim == 2:
                data = np.expand_dims(data, -1)

            if self.preprocessor:
                data = self.preprocessor(data)
            else:
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            
            tiles = self._create_tiles(data, gt, h, w, file_info['name'])
            
            for tile in tiles:
                tile['dataset'] = Path(file_info['file']).parent.name
                tile['source_file'] = file_info['name']
                samples.append(tile)
        
        np.random.seed(42)
        np.random.shuffle(samples)
        
        n_samples = len(samples)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)
        
        if self.split == 'train':
            split_samples = samples[:n_train]
        elif self.split == 'val':
            split_samples = samples[n_train:n_train + n_val]
        else:
            split_samples = samples[n_train + n_val:]
        
        if self.test_mode:
            return split_samples[:10]
        return split_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        data = sample['data'].copy()
        gt = sample['gt'].copy()
        
        if self.transform and self.split == 'train':
            data, gt = self.transform(data, gt)
        
        data_tensor = torch.from_numpy(data.transpose(2, 0, 1).copy()).float() # (C, H, W)
        gt_tensor = torch.from_numpy(gt.copy()).float()
        
        # Pad channels to max_bands
        if self.max_bands > 0:
            c, h, w = data_tensor.shape
            if c < self.max_bands:
                padding_size = self.max_bands - c
                # Pad at the end of the channel dimension
                data_tensor = F.pad(data_tensor, (0, 0, 0, 0, 0, padding_size), "constant", 0)

        return {
            'image': data_tensor,
            'gt': gt_tensor,
            'name': f"{sample['dataset']}_{sample['source_file']}"
        }


def get_dataloaders(batch_size: int = 8,
                   data_dir: str = './data',
                   num_workers: int = 4,
                   test_mode: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for HSI data."""
    try:
        from .preprocessing import HyperspectralPreprocessor, DataAugmentation
        preprocessor = HyperspectralPreprocessor(normalize_method='minmax')
        augmentor = DataAugmentation(flip_prob=0.5, rotate_prob=0.3)
    except ImportError:
        preprocessor, augmentor = None, None
        print("Warning: Preprocessing module not found, using basic normalization.")
    
    train_dataset = HyperspectralDataset(
        data_dir=data_dir, split='train', transform=augmentor,
        preprocessor=preprocessor, test_mode=test_mode
    )
    val_dataset = HyperspectralDataset(
        data_dir=data_dir, split='val', transform=None,
        preprocessor=preprocessor, test_mode=test_mode
    )
    
    if len(train_dataset) == 0:
        print("HSI training dataset is empty. Cannot create DataLoader.")
        train_loader = None
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
    
    if len(val_dataset) == 0:
        print("HSI validation dataset is empty. Cannot create DataLoader.")
        val_loader = None
    else:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    if train_loader and val_loader:
        print(f"HSI DataLoaders created: {len(train_loader)} train batches, {len(val_loader)} val batches.")
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing HSI data loader...")
    project_root = Path(__file__).parent.parent
    train_loader, val_loader = get_dataloaders(batch_size=2, data_dir=str(project_root / 'data'), test_mode=True)
    
    if train_loader and len(train_loader) > 0:
        try:
            for i, batch in enumerate(train_loader):
                print(f"Successfully loaded batch {i+1}!")
                print(f"  Image batch shape: {batch['image'].shape}")
                print(f"  GT batch shape: {batch['gt'].shape}")
                if i == 0: # Check one batch is enough
                    break
        except Exception as e:
            print(f"\nERROR during batch loading: {e}")
            print("This likely means there is still a data shape mismatch.")
    else:
        print("\nCould not load a batch from HSI train_loader.")
