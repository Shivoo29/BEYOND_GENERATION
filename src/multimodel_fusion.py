import torch
import torch.nn as nn
from src.stage2_ebt import EnergyBasedTransformer
from src.thermal_model import ThermalAnomalyDetector
from src.config import config

class MultimodalEnergyFunction(nn.Module):
    """
    Joint energy function for hyperspectral-thermal fusion
    This is the core innovation of your paper - treating multimodal
    fusion as unified compatibility evaluation rather than feature concatenation
    """
    
    def __init__(self):
        super().__init__()
        
        # Individual modality energy functions
        self.hsi_energy_fn = EnergyBasedTransformer()
        self.tir_energy_fn = ThermalAnomalyDetector(mode='energy')
        
        # Cross-modal consistency network
        self.hsi_feature_extractor = nn.Sequential(
            nn.Conv2d(config.embed_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1)
        )
        
        self.tir_feature_extractor = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1)
        )
        
        # Alignment network to project features into common space
        self.alignment = nn.Linear(64, 64)
        
        # Learnable weights for energy combination
        self.w_hsi = nn.Parameter(torch.tensor(1.0))
        self.w_tir = nn.Parameter(torch.tensor(1.0))
        self.w_cross = nn.Parameter(torch.tensor(0.5))
        
    def cross_modal_energy(self, X_hsi, T_tir, A):
        """
        Compute cross-modal consistency energy
        This measures whether HSI and TIR provide consistent evidence
        
        The key idea: if both modalities independently suggest an anomaly
        in the same location, that detection should have lower energy
        than if only one modality suggests it
        """
        # Extract features from each modality
        # Note: This requires modifying the base models to return intermediate features
        hsi_features = self.hsi_feature_extractor(X_hsi)  # Placeholder
        tir_features = self.tir_feature_extractor(T_tir)  # Placeholder
        
        # Align TIR features to HSI space
        B, C, H, W = hsi_features.shape
        tir_aligned = self.alignment(
            tir_features.permute(0, 2, 3, 1).reshape(B * H * W, C)
        ).reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Compute feature similarity
        similarity = torch.cosine_similarity(hsi_features, tir_aligned, dim=1)
        
        # Energy is low when anomalies correspond to high cross-modal agreement
        cross_energy = -(A * similarity).mean()
        
        return cross_energy
    
    def forward(self, X_hsi, S_hsi, T_tir, A):
        """
        Compute joint energy E(X_hsi, T_tir, A)
        
        This is the complete multimodal energy function that integrates:
        1. Hyperspectral evidence through spectral-spatial analysis
        2. Thermal evidence through temperature deviation analysis
        3. Cross-modal consistency through feature alignment
        
        Args:
            X_hsi: hyperspectral data (B, H, W, C)
            S_hsi: sparse component from LRSR (B, H, W, C)
            T_tir: thermal infrared data (B, 1, H, W)
            A: candidate anomaly map (B, H, W)
            
        Returns:
            total_energy: scalar value measuring configuration compatibility
        """
        # Compute individual modality energies
        E_hsi = self.hsi_energy_fn(X_hsi, S_hsi, A)
        E_tir = self.tir_energy_fn(T_tir, A)
        
        # Compute cross-modal consistency energy
        E_cross = self.cross_modal_energy(X_hsi, T_tir, A)
        
        # Weighted combination
        total_energy = (
            self.w_hsi * E_hsi + 
            self.w_tir * E_tir + 
            self.w_cross * E_cross
        )
        
        return total_energy

class MultimodalInference:
    """
    Inference pipeline for joint HSI-TIR anomaly detection
    Uses iterative energy minimization on the joint energy function
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
    @torch.no_grad()
    def joint_refinement(self, X_hsi, S_hsi, T_tir, num_iters=None):
        """
        Iteratively refine anomaly map using both modalities
        
        The gradient naturally integrates evidence from both sources:
        - Regions suggested by both HSI and TIR receive reinforcing gradients
        - Regions suggested by only one modality show higher uncertainty
        - The optimization converges to configurations consistent with both
        """
        num_iters = num_iters or config.num_refinement_iters
        
        # Initialize from HSI sparse component
        from src.stage1_lrsr import LRSR
        lrsr = LRSR()
        _, S = lrsr(X_hsi.squeeze(0).cpu().numpy())
        
        S_norm = torch.norm(torch.from_numpy(S), dim=-1)
        threshold = S_norm.mean() + 2 * S_norm.std()
        A = (S_norm > threshold).float().to(self.device)
        A.requires_grad = True
        
        # Refinement loop with joint energy
        energies = []
        for t in range(num_iters):
            # Compute joint energy
            energy = self.model(X_hsi, S_hsi, T_tir, A.squeeze(0))
            energies.append(energy.item())
            
            # Compute gradient incorporating both modalities
            energy.backward()
            grad = A.grad
            
            # Adaptive step size
            alpha = config.initial_step_size * (1 - t / num_iters) ** config.step_decay_power
            
            # Langevin dynamics
            noise = torch.randn_like(A) * (2 * alpha * config.langevin_noise) ** 0.5
            
            # Update
            with torch.no_grad():
                A = A - alpha * grad + noise
                A = torch.clamp(A, 0, 1)
            
            A.requires_grad = True
            
            # Early stopping
            if t > 0 and abs(energies[-1] - energies[-2]) < config.convergence_threshold:
                print(f"Converged at iteration {t}")
                break
        
        return A.squeeze(0).detach().cpu().numpy(), energies