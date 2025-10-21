import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config

class SpectralEncoder(nn.Module):
    """Encode hyperspectral signatures"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: (B, H, W, C) -> (B, C, H*W)
        B, H, W, C = x.shape
        x = x.view(B, H * W, C).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x.transpose(1, 2).view(B, H, W, -1)

class SpatialEncoder(nn.Module):
    """Encode spatial patterns"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x.permute(0, 2, 3, 1)

class TransformerEncoder(nn.Module):
    """Transformer for spatial-spectral reasoning"""
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        # x: (B, H, W, C) -> (B, H*W, C)
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        x = self.transformer(x)
        return x.view(B, H, W, C)

class EnergyBasedTransformer(nn.Module):
    """Complete Energy-Based Transformer for anomaly detection"""
    
    def __init__(self):
        super().__init__()
        
        # Encoders
        self.spectral_encoder = SpectralEncoder(config.num_spectral_bands, config.embed_dim // 2)
        self.spatial_encoder = SpatialEncoder(config.num_spectral_bands, config.embed_dim // 2)
        self.anomaly_encoder = nn.Conv2d(1, config.embed_dim // 4, kernel_size=1)
        
        # Transformer
        self.transformer = TransformerEncoder(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )
        
        # Energy head
        self.energy_head = nn.Sequential(
            nn.Linear(config.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Learnable anomaly prototypes
        self.anomaly_prototypes = nn.Parameter(torch.randn(10, config.num_spectral_bands))
        
    def spectral_energy(self, X, A):
        """Compute spectral compatibility energy"""
        # X: (B, H, W, C), A: (B, H, W)
        B, H, W, C = X.shape
        X_flat = X.view(B * H * W, C)
        A_flat = A.view(B * H * W)
        
        # Compute similarity to prototypes
        similarities = torch.matmul(X_flat, self.anomaly_prototypes.t())
        max_similarity, _ = torch.max(similarities, dim=1)
        
        # Weight by anomaly map
        energy = -(A_flat * max_similarity).sum() / (A_flat.sum() + 1e-6)
        return energy
    
    def spatial_energy(self, S_encoded, A):
        """Compute spatial consistency energy"""
        # S_encoded: (B, H, W, C), A: (B, H, W)
        A_expanded = A.unsqueeze(-1)
        diff = (S_encoded - A_expanded) ** 2
        return diff.mean()
    
    def prior_energy(self, A):
        """Compute prior energy (total variation + sparsity)"""
        # Total variation
        tv_h = torch.abs(A[:, 1:, :] - A[:, :-1, :]).sum()
        tv_w = torch.abs(A[:, :, 1:] - A[:, :, :-1]).sum()
        tv = tv_h + tv_w
        
        # Sparsity penalty
        sparsity = A.sum()
        
        return config.w_prior * (tv + 0.01 * sparsity)
    
    def forward(self, X, S, A):
        """
        Compute energy E(X, S, A)
        
        Args:
            X: hyperspectral data (B, H, W, C)
            S: sparse component from LRSR (B, H, W, C)
            A: candidate anomaly map (B, H, W)
            
        Returns:
            energy: scalar energy value
        """
        # Encode inputs
        spectral_features = self.spectral_encoder(X)
        spatial_features = self.spatial_encoder(S)
        anomaly_features = self.anomaly_encoder(A.unsqueeze(1)).permute(0, 2, 3, 1)
        
        # Combine features
        combined = torch.cat([spectral_features, spatial_features, anomaly_features], dim=-1)
        
        # Transformer reasoning
        features = self.transformer(combined)
        
        # Global pooling and energy computation
        pooled = features.mean(dim=[1, 2])  # (B, C)
        base_energy = self.energy_head(pooled).squeeze()  # (B,)
        
        # Add component energies
        spectral_e = self.spectral_energy(X, A)
        spatial_e = self.spatial_energy(spatial_features, A)
        prior_e = self.prior_energy(A)
        
        total_energy = base_energy.mean() + config.w_spectral * spectral_e + config.w_spatial * spatial_e + prior_e
        
        return total_energy