from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # Data parameters
    tile_size: int = 256
    tile_overlap: int = 32
    num_spectral_bands: int = 200
    
    # LRSR parameters
    lrsr_lambda_l: float = 0.01
    lrsr_lambda_s: float = 0.1
    lrsr_mu: float = 0.1
    lrsr_max_iters: int = 100
    lrsr_tol: float = 1e-4
    
    # EBT architecture parameters
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_ratio: int = 4
    
    # Energy function weights
    w_spectral: float = 1.0
    w_spatial: float = 1.0
    w_prior: float = 0.5
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 50
    num_workers: int = 4
    
    # Inference parameters
    num_refinement_iters: int = 10
    initial_step_size: float = 0.1
    step_decay_power: float = 0.5
    langevin_noise: float = 0.01
    convergence_threshold: float = 1e-3
    
    # Paths
    data_dir: str = "./data"
    cache_dir: str = "./data/cache"
    checkpoint_dir: str = "./models/checkpoints"
    results_dir: str = "./results"
    
    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    
config = Config()