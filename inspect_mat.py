

import scipy.io
import numpy as np

try:
    mat_data_syn = scipy.io.loadmat('/home/skj/Documents/projects/Beyond_Generation/data/raw/Salinas/Salians_120_sj25_syn.mat')
    print("Salinas_syn.mat keys:", mat_data_syn.keys())
    for key, value in mat_data_syn.items():
        if isinstance(value, np.ndarray):
            print(f"  Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")

    mat_data_gt = scipy.io.loadmat('/home/skj/Documents/projects/Beyond_Generation/data/raw/Salinas/Salians_120_sj25_gt.mat')
    print("\nSalinas_gt.mat keys:", mat_data_gt.keys())
    for key, value in mat_data_gt.items():
        if isinstance(value, np.ndarray):
            print(f"  Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")

    mat_data_abu = scipy.io.loadmat('/home/skj/Documents/projects/Beyond_Generation/data/raw/abu-airport-1.mat')
    print("\nAbu-airport-1.mat keys:", mat_data_abu.keys())
    for key, value in mat_data_abu.items():
        if isinstance(value, np.ndarray):
            print(f"  Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")

except ImportError:
    print("scipy not installed. Please install it using 'pip install scipy'")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

