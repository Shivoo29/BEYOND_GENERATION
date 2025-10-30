"""
Stage 2: Energy-Based Transformer (EBT) for Anomaly Refinement
Implements verification-driven inference through iterative energy minimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config

class SpectralEncoder(nn.Module):
    """
    Encode hyperspectral signatures
    Adapts to varying input channels dynamically
    """
    def __init__(self, in_channels=None, out_channels=None):
        super().__init__()
        self.in_channels = in_channels or config.num_spectral_bands
        self.out_channels = out_channels or (config.embed_dim // 2)
        
        self.conv1 = nn.Conv1d(self.in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, self.out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        
        # Channel adaptation layer (initialized but may be replaced)
        self.adapt_layer = None
        
    def forward(self, x):
        # x: (B, H, W, C) -> (B, C, H*W)
        B, H, W, C = x.shape
        
        # Adapt to actual input channels if needed
        if C != self.in_channels:
            if self.adapt_layer is None or self.adapt_layer.in_channels != C:
                self.adapt_layer = nn.Conv1d(C, self.in_channels, kernel_size=1).to(x.device)
                print(f"SpectralEncoder: Adapted to {C} input channels (expected {self.in_channels})")
            x = x.reshape(B, H * W, C).transpose(1, 2)  # (B, C, H*W)
            x = self.adapt_layer(x)  # (B, in_channels, H*W)
        else:
            x = x.reshape(B, H * W, C).transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x.transpose(1, 2).reshape(B, H, W, -1)  # (B, H, W, out_channels)


class SpatialEncoder(nn.Module):
    """
    Encode spatial patterns
    Adapts to varying input channels dynamically
    """
    def __init__(self, in_channels=None, out_channels=None):
        super().__init__()
        self.in_channels = in_channels or config.num_spectral_bands
        self.out_channels = out_channels or (config.embed_dim // 2)
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, self.out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        
        # Channel adaptation layer
        self.adapt_layer = None
        
    def forward(self, x):
        # x: (B, H, W, C) -> (B, C, H, W)
        B, H, W, C = x.shape
        
        # Adapt to actual input channels if needed
        if C != self.in_channels:
            if self.adapt_layer is None or self.adapt_layer.in_channels != C:
                self.adapt_layer = nn.Conv2d(C, self.in_channels, kernel_size=1).to(x.device)
                print(f"SpatialEncoder: Adapted to {C} input channels (expected {self.in_channels})")
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.adapt_layer(x)  # (B, in_channels, H, W)
        else:
            x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x.permute(0, 2, 3, 1)  # (B, H, W, out_channels)


class TransformerEncoder(nn.Module):
    """
    Transformer for spatial-spectral reasoning
    Processes features through multi-head self-attention
    """
    def __init__(self, embed_dim=None, num_heads=None, num_layers=None):
        super().__init__()
        self.embed_dim = embed_dim or config.embed_dim
        self.num_heads = num_heads or config.num_heads
        self.num_layers = num_layers or config.num_layers
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
    def forward(self, x):
        # x: (B, H, W, C) -> (B, H*W, C)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = self.transformer(x)
        return x.reshape(B, H, W, C)


class EnergyBasedTransformer(nn.Module):
    """
    Complete Energy-Based Transformer for anomaly detection
    Implements verification-driven inference through energy minimization
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoders - each outputs embed_dim // 2 features
        self.spectral_encoder = SpectralEncoder(
            in_channels=config.num_spectral_bands,
            out_channels=config.embed_dim // 2
        )
        self.spatial_encoder = SpatialEncoder(
            in_channels=config.num_spectral_bands,
            out_channels=config.embed_dim // 2
        )
        
        # Anomaly encoder - outputs embed_dim // 4 features
        self.anomaly_encoder = nn.Sequential(
            nn.Conv2d(1, config.embed_dim // 4, kernel_size=1),
            nn.BatchNorm2d(config.embed_dim // 4),
            nn.ReLU()
        )
        
        # Combined features dimension: (embed_dim//2) + (embed_dim//2) + (embed_dim//4) 
        # = embed_dim + embed_dim//4 = 5*embed_dim//4
        combined_dim = config.embed_dim + config.embed_dim // 4
        
        # Projection layer to match transformer d_model
        self.feature_projection = nn.Linear(combined_dim, config.embed_dim)
        
        # Transformer
        self.transformer = TransformerEncoder(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )
        
        # Energy head - maps features to scalar energy
        self.energy_head = nn.Sequential(
            nn.Linear(config.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Learnable anomaly prototypes for spectral energy
        self.anomaly_prototypes = nn.Parameter(
            torch.randn(10, config.num_spectral_bands)
        )
        
        # Adaptive projection for spectral energy (handles varying band counts)
        self.spectral_projection = None
        
    def spectral_energy(self, X, A):
        """
        Compute spectral compatibility energy
        Low energy when anomalies have characteristic spectral signatures
        """
        B, H, W, C = X.shape
        X_flat = X.reshape(B * H * W, C)
        A_flat = A.reshape(B * H * W)
        
        # Adapt to actual input channels if needed
        if C != self.anomaly_prototypes.shape[1]:
            if self.spectral_projection is None or self.spectral_projection.in_features != C:
                self.spectral_projection = nn.Linear(C, self.anomaly_prototypes.shape[1]).to(X.device)
                # print(f"SpectralEnergy: Adapted to {C} input channels (expected {self.anomaly_prototypes.shape[1]})")
            X_flat = self.spectral_projection(X_flat)
        
        # Compute similarity to learned anomaly prototypes
        similarities = torch.matmul(X_flat, self.anomaly_prototypes.t())
        max_similarity, _ = torch.max(similarities, dim=1)
        
        # Weight by anomaly map - negative because high similarity should give low energy
        energy = -(A_flat * max_similarity).sum() / (A_flat.sum() + 1e-6)
        return energy * config.w_spectral
    
    def spatial_energy(self, S_encoded, A):
        """
        Compute spatial consistency energy
        Low energy when anomalies align with spatial features
        """
        A_expanded = A.unsqueeze(-1)
        diff = (S_encoded - A_expanded) ** 2
        return diff.mean() * config.w_spatial
    
    def prior_energy(self, A):
        """
        Compute prior energy (total variation + sparsity)
        Encourages smooth and sparse anomaly maps
        """
        # Total variation for spatial smoothness
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
            energy: scalar energy value (lower is better)
        """
        # Encode inputs
        spectral_features = self.spectral_encoder(X)  # (B, H, W, embed_dim//2)
        spatial_features = self.spatial_encoder(S)     # (B, H, W, embed_dim//2)
        
        # Encode anomaly map
        A_4d = A.unsqueeze(1)  # (B, 1, H, W)
        anomaly_features = self.anomaly_encoder(A_4d).permute(0, 2, 3, 1)  # (B, H, W, embed_dim//4)
        
        # Combine features
        combined = torch.cat([spectral_features, spatial_features, anomaly_features], dim=-1)
        # combined shape: (B, H, W, 5*embed_dim//4)
        
        # Project to transformer dimension
        combined = self.feature_projection(combined)  # (B, H, W, embed_dim)
        
        # Transformer reasoning
        features = self.transformer(combined)  # (B, H, W, embed_dim)
        
        # Global pooling and energy computation
        pooled = features.mean(dim=[1, 2])  # (B, embed_dim)
        base_energy = self.energy_head(pooled).squeeze()  # (B,)
        
        # Add component energies
        spectral_e = self.spectral_energy(X, A)
        spatial_e = self.spatial_energy(spatial_features, A)
        prior_e = self.prior_energy(A)
        
        # Total energy
        total_energy = base_energy.mean() + spectral_e + spatial_e + prior_e
        
        return total_energy


# Iterative refinement for inference
class EBTInference:
    """
    Inference pipeline with iterative energy minimization
    Implements System 2 thinking through verification
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
    @torch.no_grad()
    def iterative_refinement(self, X, S, num_iters=10):
        """
        Iteratively refine anomaly map through energy minimization
        
        Args:
            X: hyperspectral data tensor (1, H, W, C)
            S: sparse component tensor (1, H, W, C)
            num_iters: number of refinement iterations
            
        Returns:
            A: refined anomaly map (H, W)
        """
        # Initialize from thresholded sparse component
        S_norm = torch.norm(S, dim=-1)
        threshold = S_norm.mean() + 2 * S_norm.std()
        A = (S_norm > threshold).float()
        A = A.to(self.device)
        A.requires_grad = True
        
        # Refinement loop
        prev_energy = float('inf')
        for t in range(num_iters):
            # Compute energy
            energy = self.model(X, S, A.squeeze(0))
            
            # Compute gradient
            energy.backward()
            grad = A.grad
            
            # Update step size (decay over iterations)
            alpha = config.initial_step_size * (1 - t / num_iters) ** config.step_decay_power
            
            # Add Langevin noise for exploration
            noise = torch.randn_like(A) * (2 * alpha * config.langevin_noise) ** 0.5
            
            # Update anomaly map
            with torch.no_grad():
                A = A - alpha * grad + noise
                A = torch.clamp(A, 0, 1)
            
            A.requires_grad = True
            
            # Check convergence
            if t > 0 and abs(energy.item() - prev_energy) < config.convergence_threshold:
                print(f"Converged at iteration {t}")
                break
            prev_energy = energy.item()
        
        return A.squeeze(0).detach().cpu().numpy()