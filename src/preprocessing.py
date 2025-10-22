"""
Data preprocessing utilities for hyperspectral and thermal imagery
Handles normalization, augmentation, and band selection
"""

import numpy as np
import torch
from scipy import ndimage
from typing import Tuple, Optional
import cv2

class HyperspectralPreprocessor:
    """Preprocessing pipeline for hyperspectral data"""
    
    def __init__(self, 
                 normalize_method='minmax',
                 remove_water_bands=True,
                 band_selection=None):
        """
        Args:
            normalize_method: 'minmax', 'standard', or 'per_band'
            remove_water_bands: Remove noisy water absorption bands
            band_selection: List of band indices to keep, None for all
        """
        self.normalize_method = normalize_method
        self.remove_water_bands = remove_water_bands
        self.band_selection = band_selection
        self.water_bands = [104, 105, 106, 113, 114, 115, 116, 117, 118, 119, 
                           120, 153, 154, 155, 156, 157, 158, 159, 160, 161, 
                           162, 163, 164, 165, 166]  # Common water absorption bands
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize hyperspectral data
        
        Args:
            data: (H, W, B) hyperspectral image
            
        Returns:
            Normalized data
        """
        if self.normalize_method == 'minmax':
            # Global min-max normalization
            data_min = data.min()
            data_max = data.max()
            return (data - data_min) / (data_max - data_min + 1e-8)
            
        elif self.normalize_method == 'standard':
            # Z-score normalization
            mean = data.mean()
            std = data.std()
            return (data - mean) / (std + 1e-8)
            
        elif self.normalize_method == 'per_band':
            # Normalize each band independently
            H, W, B = data.shape
            normalized = np.zeros_like(data)
            for b in range(B):
                band = data[:, :, b]
                band_min = band.min()
                band_max = band.max()
                normalized[:, :, b] = (band - band_min) / (band_max - band_min + 1e-8)
            return normalized
            
        return data
    
    def remove_noisy_bands(self, data: np.ndarray) -> np.ndarray:
        """Remove water absorption and other noisy bands"""
        if not self.remove_water_bands:
            return data
            
        H, W, B = data.shape
        valid_bands = [i for i in range(B) if i not in self.water_bands]
        return data[:, :, valid_bands]
    
    def select_bands(self, data: np.ndarray) -> np.ndarray:
        """Select specific bands if specified"""
        if self.band_selection is None:
            return data
        return data[:, :, self.band_selection]
    
    def pca_reduction(self, data: np.ndarray, n_components: int = 30) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            data: (H, W, B) hyperspectral image
            n_components: Number of principal components to keep
            
        Returns:
            Reduced data (H, W, n_components)
        """
        H, W, B = data.shape
        data_flat = data.reshape(H * W, B)
        
        # Center the data
        mean = data_flat.mean(axis=0)
        centered = data_flat - mean
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx[:n_components]]
        
        # Project data
        projected = centered @ eigenvectors
        
        return projected.reshape(H, W, n_components)
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline"""
        data = self.remove_noisy_bands(data)
        data = self.select_bands(data)
        data = self.normalize(data)
        return data


class DataAugmentation:
    """Augmentation strategies for hyperspectral and thermal data"""
    
    def __init__(self, 
                 flip_prob=0.5,
                 rotate_prob=0.3,
                 noise_prob=0.2,
                 spectral_shift_prob=0.3):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.spectral_shift_prob = spectral_shift_prob
    
    def random_flip(self, image: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random horizontal and vertical flips"""
        if np.random.random() < self.flip_prob:
            if np.random.random() < 0.5:
                image = np.fliplr(image)
                gt = np.fliplr(gt)
            else:
                image = np.flipud(image)
                gt = np.flipud(gt)
        return image, gt
    
    def random_rotation(self, image: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random 90-degree rotations"""
        if np.random.random() < self.rotate_prob:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            image = np.rot90(image, k)
            gt = np.rot90(gt, k)
        return image, gt
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to hyperspectral data"""
        if np.random.random() < self.noise_prob:
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.randn(*image.shape) * noise_level
            image = image + noise
            image = np.clip(image, 0, 1)
        return image
    
    def spectral_shift(self, image: np.ndarray) -> np.ndarray:
        """Simulate spectral variability"""
        if len(image.shape) == 3 and np.random.random() < self.spectral_shift_prob:
            # Random scaling per band
            H, W, B = image.shape
            scale = np.random.uniform(0.9, 1.1, (1, 1, B))
            image = image * scale
            image = np.clip(image, 0, 1)
        return image
    
    def random_crop(self, image: np.ndarray, gt: np.ndarray, 
                    crop_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Random spatial cropping"""
        H, W = image.shape[:2]
        if H > crop_size and W > crop_size:
            h_start = np.random.randint(0, H - crop_size)
            w_start = np.random.randint(0, W - crop_size)
            
            if len(image.shape) == 3:
                image = image[h_start:h_start+crop_size, w_start:w_start+crop_size, :]
            else:
                image = image[h_start:h_start+crop_size, w_start:w_start+crop_size]
            
            gt = gt[h_start:h_start+crop_size, w_start:w_start+crop_size]
        
        return image, gt
    
    def __call__(self, image: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation pipeline"""
        image, gt = self.random_flip(image, gt)
        image, gt = self.random_rotation(image, gt)
        image = self.add_noise(image)
        image = self.spectral_shift(image)
        return image, gt


class ThermalPreprocessor:
    """Preprocessing for thermal infrared imagery"""
    
    def __init__(self, target_size=None):
        self.target_size = target_size
    
    def landsat_to_temperature(self, dn_values: np.ndarray, 
                               band_number: int = 10,
                               metadata: dict = None) -> np.ndarray:
        """
        Convert Landsat thermal band DN values to brightness temperature
        
        Args:
            dn_values: Digital number values from Landsat
            band_number: Thermal band number (10 or 11 for Landsat 8/9)
            metadata: Calibration coefficients from metadata
            
        Returns:
            Temperature in Celsius
        """
        # Default calibration values for Landsat 8 band 10
        if metadata is None:
            metadata = {
                'RADIANCE_MULT': 3.3420E-04,
                'RADIANCE_ADD': 0.10000,
                'K1_CONSTANT': 774.8853,
                'K2_CONSTANT': 1321.0789
            }
        
        # Step 1: Convert DN to radiance
        radiance = metadata['RADIANCE_MULT'] * dn_values + metadata['RADIANCE_ADD']
        
        # Step 2: Convert radiance to brightness temperature (Kelvin)
        temp_kelvin = metadata['K2_CONSTANT'] / np.log((metadata['K1_CONSTANT'] / radiance) + 1)
        
        # Step 3: Convert to Celsius
        temp_celsius = temp_kelvin - 273.15
        
        return temp_celsius
    
    def normalize_temperature(self, temp: np.ndarray) -> np.ndarray:
        """Normalize temperature to [0, 1] range"""
        # Remove outliers
        p_low, p_high = np.percentile(temp, [2, 98])
        temp_clipped = np.clip(temp, p_low, p_high)
        
        # Normalize
        temp_norm = (temp_clipped - temp_clipped.min()) / (temp_clipped.max() - temp_clipped.min() + 1e-8)
        
        return temp_norm
    
    def resize(self, image: np.ndarray, gt: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resize thermal image to target size"""
        if self.target_size is None:
            return image, gt
            
        image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        if gt is not None:
            gt_resized = cv2.resize(gt, self.target_size, interpolation=cv2.INTER_NEAREST)
            return image_resized, gt_resized
            
        return image_resized, None
    
    def __call__(self, thermal: np.ndarray, gt: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply thermal preprocessing pipeline"""
        thermal = self.normalize_temperature(thermal)
        thermal, gt = self.resize(thermal, gt)
        return thermal, gt


def prepare_training_data(hsi_path: str, gt_path: str, 
                          preprocessor: HyperspectralPreprocessor,
                          augmentor: DataAugmentation = None,
                          tile_size: int = 256,
                          stride: int = 128) -> list:
    """
    Prepare training data by tiling large images
    
    Args:
        hsi_path: Path to hyperspectral image
        gt_path: Path to ground truth
        preprocessor: Preprocessing pipeline
        augmentor: Augmentation pipeline
        tile_size: Size of tiles to extract
        stride: Stride for tile extraction (overlap = tile_size - stride)
        
    Returns:
        List of (tile, gt_tile) pairs
    """
    # Load data
    hsi = np.load(hsi_path) if hsi_path.endswith('.npy') else load_mat_file(hsi_path)
    gt = np.load(gt_path) if gt_path.endswith('.npy') else load_mat_file(gt_path)
    
    # Preprocess
    hsi = preprocessor(hsi)
    
    # Extract tiles
    tiles = []
    H, W = hsi.shape[:2]
    
    for h in range(0, H - tile_size + 1, stride):
        for w in range(0, W - tile_size + 1, stride):
            if len(hsi.shape) == 3:
                hsi_tile = hsi[h:h+tile_size, w:w+tile_size, :]
            else:
                hsi_tile = hsi[h:h+tile_size, w:w+tile_size]
                
            gt_tile = gt[h:h+tile_size, w:w+tile_size]
            
            # Skip tiles with no anomalies (optional)
            if gt_tile.sum() > 0:
                # Apply augmentation if specified
                if augmentor is not None:
                    hsi_tile, gt_tile = augmentor(hsi_tile, gt_tile)
                
                tiles.append((hsi_tile, gt_tile))
    
    return tiles


def load_mat_file(path: str) -> np.ndarray:
    """Load data from .mat file"""
    import scipy.io
    mat_data = scipy.io.loadmat(path)
    
    # Find the main data variable (usually the largest array)
    for key, value in mat_data.items():
        if isinstance(value, np.ndarray) and not key.startswith('__'):
            return value
    
    raise ValueError(f"Could not find data array in {path}")