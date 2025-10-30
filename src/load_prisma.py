#!/usr/bin/env python
"""
Quick fix for PRISMA .he5 files
Add this function to your test_hsi.py or run standalone
"""

import h5py
import numpy as np

def load_prisma_he5(filepath):
    """
    Load PRISMA L2D .he5 file and extract hyperspectral data
    
    Returns:
        data: numpy array (H, W, C) - combined VNIR+SWIR
        metadata: dict with file info
    """
    print(f"Loading PRISMA file: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        swaths = f['HDFEOS']['SWATHS']
        
        # Get VNIR and SWIR cubes from PRS_L2D_HCO
        if 'PRS_L2D_HCO' in swaths:
            data_fields = swaths['PRS_L2D_HCO']['Data Fields']
            
            # Load VNIR (66 bands)
            vnir = data_fields['VNIR_Cube'][:]  # (1186, 66, 1196)
            print(f"  VNIR shape: {vnir.shape}")
            
            # Load SWIR (173 bands)
            swir = data_fields['SWIR_Cube'][:]  # (1186, 173, 1196)
            print(f"  SWIR shape: {swir.shape}")
            
            # Transpose to (H, W, C) format
            vnir = np.transpose(vnir, (0, 2, 1))  # (1186, 1196, 66)
            swir = np.transpose(swir, (0, 2, 1))  # (1186, 1196, 173)
            
            # Combine VNIR + SWIR
            data = np.concatenate([vnir, swir], axis=2)  # (1186, 1196, 239)
            print(f"  Combined HSI shape: {data.shape}")
            
            metadata = {
                'filename': filepath.split('/')[-1],
                'n_bands': data.shape[2],
                'vnir_bands': 66,
                'swir_bands': 173,
                'sensor': 'PRISMA'
            }
            
            return data, metadata
        
        else:
            raise ValueError(f"PRS_L2D_HCO swath not found in {filepath}")

# Test it
if __name__ == "__main__":
    filepath = "data/processed/mock_dataset_15_oct/Hypersepctral Anomaly Datasets/PRS_L2D_STD_20201214060713_20201214060717_0001.he5"
    
    data, metadata = load_prisma_he5(filepath)
    
    print("\n=== Data Info ===")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Min: {data.min()}, Max: {data.max()}")
    print(f"Bands: {metadata['n_bands']}")
    
    # Save for testing
    print("\nSaving as numpy file for quick testing...")
    np.save('prisma_extracted.npy', data)
    print("Saved as: prisma_extracted.npy")