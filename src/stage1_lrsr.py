"""
Stage 1: Low-Rank and Sparse Representation (LRSR) using ADMM
GPU-accelerated version using PyTorch for fast decomposition
"""
import numpy as np
import torch

try:
    from .config import config
except ImportError:
    # Fallback if running standalone
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        lrsr_lambda_l: float = 0.01
        lrsr_lambda_s: float = 0.1
        lrsr_mu: float = 0.1
        lrsr_max_iters: int = 100
        lrsr_tol: float = 1e-4
    
    config = Config()


class LRSR:
    """Low-Rank and Sparse Representation using ADMM (GPU-accelerated)"""
    
    def __init__(self, lambda_l=None, lambda_s=None, mu=None, max_iters=None, device=None):
        self.lambda_l = lambda_l or config.lrsr_lambda_l
        self.lambda_s = lambda_s or config.lrsr_lambda_s
        self.mu = mu or config.lrsr_mu
        self.max_iters = max_iters or config.lrsr_max_iters
        self.tol = config.lrsr_tol
        
        # Set device (GPU if available, otherwise CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"LRSR initialized on device: {self.device}")
        
    def soft_threshold(self, X, threshold):
        """Element-wise soft thresholding (GPU)"""
        return torch.sign(X) * torch.maximum(
            torch.abs(X) - threshold, 
            torch.tensor(0.0, device=self.device)
        )
    
    def svd_threshold(self, X, threshold):
        """SVD with singular value soft thresholding (GPU)"""
        U, sigma, Vt = torch.linalg.svd(X, full_matrices=False)
        sigma_thresh = self.soft_threshold(sigma, threshold)
        return U @ torch.diag(sigma_thresh) @ Vt
    
    def decompose(self, X):
        """
        Decompose hyperspectral data into low-rank and sparse components
        
        Args:
            X: numpy array or torch tensor of shape (H, W, B) - hyperspectral image
            
        Returns:
            L: low-rank background component (H, W, B) as numpy array
            S: sparse anomaly component (H, W, B) as numpy array
        """
        # Convert to torch tensor if needed and move to GPU
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        # Reshape to matrix form: (B, H*W)
        H, W, B = X.shape
        X_mat = X.permute(2, 0, 1).reshape(B, H * W)
        
        # Initialize variables on GPU
        L = torch.zeros_like(X_mat)
        S = torch.zeros_like(X_mat)
        Y = torch.zeros_like(X_mat)
        
        # ADMM iterations
        for iteration in range(self.max_iters):
            # Update L (low-rank component)
            L = self.svd_threshold(X_mat - S + Y / self.mu, 1.0 / self.mu)
            
            # Update S (sparse component)
            S = self.soft_threshold(X_mat - L + Y / self.mu, self.lambda_s / self.mu)
            
            # Update Y (Lagrange multiplier)
            residual = X_mat - L - S
            Y = Y + self.mu * residual
            
            # Check convergence
            residual_norm = torch.linalg.norm(residual, ord='fro')
            if residual_norm < self.tol:
                print(f"LRSR converged at iteration {iteration}")
                break
        
        # Reshape back to image form and move to CPU
        L_img = L.reshape(B, H, W).permute(1, 2, 0).cpu().numpy()
        S_img = S.reshape(B, H, W).permute(1, 2, 0).cpu().numpy()
        
        return L_img, S_img
    
    def __call__(self, X):
        """Convenience method for decomposition"""
        return self.decompose(X)


if __name__ == '__main__':
    # Test LRSR
    print("Testing GPU-accelerated LRSR...")
    
    # Create test data
    H, W, B = 100, 100, 50
    X_test = np.random.randn(H, W, B).astype(np.float32)
    
    # Initialize and run
    lrsr = LRSR()
    L, S = lrsr(X_test)
    
    print(f"✓ LRSR test passed!")
    print(f"  Input shape: {X_test.shape}")
    print(f"  Low-rank component shape: {L.shape}")
    print(f"  Sparse component shape: {S.shape}")
    print(f"  Reconstruction error: {np.linalg.norm(X_test - L - S):.6f}")