import numpy as np
import torch
from scipy.linalg import svd
from src.config import config

class LRSR:
    """Low-Rank and Sparse Representation using ADMM"""
    
    def __init__(self, lambda_l=None, lambda_s=None, mu=None, max_iters=None):
        self.lambda_l = lambda_l or config.lrsr_lambda_l
        self.lambda_s = lambda_s or config.lrsr_lambda_s
        self.mu = mu or config.lrsr_mu
        self.max_iters = max_iters or config.lrsr_max_iters
        self.tol = config.lrsr_tol
        
    def soft_threshold(self, X, threshold):
        """Element-wise soft thresholding"""
        return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)
    
    def svd_threshold(self, X, threshold):
        """SVD with singular value soft thresholding"""
        U, sigma, Vt = svd(X, full_matrices=False)
        sigma_thresh = self.soft_threshold(sigma, threshold)
        return U @ np.diag(sigma_thresh) @ Vt
    
    def decompose(self, X):
        """
        Decompose hyperspectral data into low-rank and sparse components
        
        Args:
            X: numpy array of shape (H, W, B) - hyperspectral image
            
        Returns:
            L: low-rank background component (H, W, B)
            S: sparse anomaly component (H, W, B)
        """
        # Reshape to matrix form: (B, H*W)
        H, W, B = X.shape
        X_mat = X.transpose(2, 0, 1).reshape(B, H * W)
        
        # Initialize variables
        L = np.zeros_like(X_mat)
        S = np.zeros_like(X_mat)
        Y = np.zeros_like(X_mat)
        
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
            residual_norm = np.linalg.norm(residual, 'fro')
            if residual_norm < self.tol:
                print(f"LRSR converged at iteration {iteration}")
                break
        
        # Reshape back to image form
        L_img = L.reshape(B, H, W).transpose(1, 2, 0)
        S_img = S.reshape(B, H, W).transpose(1, 2, 0)
        
        return L_img, S_img
    
    def __call__(self, X):
        """Convenience method for decomposition"""
        return self.decompose(X)