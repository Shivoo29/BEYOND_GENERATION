from dataclesses import dataclass
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

    # EBT architecture parameterd
    embed_dim: int = 256
    num_head:int = 8
    num_layers: int = 6
    mlp_ratio: int = 4
    