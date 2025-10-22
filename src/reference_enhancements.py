"""
Reference Code Extractor and Integrator
Extracts useful components from reference repositories without overwhelming context
"""

import os
from pathlib import Path
import ast
import re

class ReferenceExtractor:
    """Extract and integrate useful code from reference repositories"""
    
    def __init__(self, references_dir='references'):
        self.references_dir = Path(references_dir)
        self.useful_components = {}
    
    def analyze_references(self):
        """Analyze reference repos and identify useful components"""
        
        # Key methods from each reference repo based on your structure
        reference_insights = {
            'ade': {
                'description': 'Attribute profiles and morphological detection',
                'useful_files': ['morph_detect.m', 'EMAP_xdk.m', 'RRX_detect.m'],
                'key_concepts': [
                    'Morphological attribute profiles for spatial feature extraction',
                    'Extended multi-attribute profiles (EMAP)',
                    'Reed-Xiaoli detector (RX) - classic anomaly detection baseline'
                ],
                'integration': """
                # Morphological operations for post-processing (Python equivalent)
                from scipy import ndimage
                
                def morphological_cleanup(anomaly_map, iterations=2):
                    # Opening to remove small false positives
                    cleaned = ndimage.binary_opening(anomaly_map > 0.5, iterations=iterations)
                    # Closing to fill small gaps
                    cleaned = ndimage.binary_closing(cleaned, iterations=iterations)
                    return cleaned.astype(np.float32)
                """
            },
            
            'crdbpsw': {
                'description': 'Collaborative representation-based detector with sliding window',
                'useful_files': ['func_CRDBPSW.m'],
                'key_concepts': [
                    'Sliding window for local anomaly detection',
                    'Collaborative representation using background dictionary',
                    'Pseudo-inverse for fast computation'
                ],
                'integration': """
                # Sliding window detector enhancement
                def sliding_window_refinement(anomaly_scores, window_size=15):
                    from scipy.ndimage import uniform_filter
                    # Local normalization within sliding window
                    local_mean = uniform_filter(anomaly_scores, size=window_size)
                    local_std = np.sqrt(uniform_filter(anomaly_scores**2, size=window_size) - local_mean**2)
                    normalized = (anomaly_scores - local_mean) / (local_std + 1e-8)
                    return normalized
                """
            },
            
            'glrt': {
                'description': 'Generalized Likelihood Ratio Test detectors',
                'useful_files': ['func_1S_GLRT.m', 'func_2S_GLRT.m'],
                'key_concepts': [
                    'Statistical hypothesis testing for anomaly detection',
                    'Single and two-step GLRT approaches',
                    'Adaptive threshold based on false alarm rate'
                ],
                'integration': """
                # GLRT-inspired statistical validation
                def glrt_validation(anomaly_scores, background_scores, alpha=0.01):
                    from scipy import stats
                    # Compute test statistic
                    mean_bg = background_scores.mean()
                    std_bg = background_scores.std()
                    z_scores = (anomaly_scores - mean_bg) / std_bg
                    # Adaptive threshold based on desired false alarm rate
                    threshold = stats.norm.ppf(1 - alpha)
                    return z_scores > threshold
                """
            },
            
            'lrsr': {
                'description': 'Low-rank and sparse representation (Python implementation)',
                'useful_files': ['LRSR.py', 'dic_constr.py'],
                'key_concepts': [
                    'Dictionary construction from background',
                    'ADMM optimization for LRSR',
                    'Sparse coding for anomaly detection'
                ],
                'integration': """
                # Enhanced LRSR with dictionary learning
                def construct_background_dictionary(X, dict_size=50):
                    from sklearn.decomposition import DictionaryLearning
                    # Learn overcomplete dictionary from background
                    dict_learner = DictionaryLearning(
                        n_components=dict_size,
                        alpha=0.1,
                        max_iter=100,
                        random_state=42
                    )
                    H, W, B = X.shape
                    X_flat = X.reshape(H*W, B)
                    dictionary = dict_learner.fit(X_flat).components_
                    return dictionary
                """
            },
            
            'hyperad': {
                'description': 'Real-time hyperspectral anomaly detection framework',
                'useful_files': ['cdlss_ad.py', 'erx.py', 'rx_bil.py'],
                'key_concepts': [
                    'Efficient RX detector variants',
                    'Band importance learning (BIL)',
                    'Real-time processing optimizations'
                ],
                'integration': """
                # Band importance weighting from hyperad
                def compute_band_importance(X, y):
                    from sklearn.ensemble import RandomForestClassifier
                    # Use RF to compute band importance
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    H, W, B = X.shape
                    X_flat = X.reshape(H*W, B)
                    y_flat = y.reshape(H*W)
                    # Train on pixels with labels
                    mask = y_flat != -1  # Assuming -1 for unlabeled
                    if mask.sum() > 0:
                        rf.fit(X_flat[mask], y_flat[mask])
                        importance = rf.feature_importances_
                        return importance
                    return np.ones(B) / B  # Equal weights if no labels
                """
            }
        }
        
        return reference_insights
    
    def generate_integration_module(self):
        """Generate a module that integrates best practices from references"""
        
        integration_code = '''"""
Enhanced components integrated from reference implementations
Use these to augment the main EBT pipeline
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage, stats
from sklearn.decomposition import DictionaryLearning
from sklearn.ensemble import RandomForestClassifier


class ReferenceEnhancements:
    """Collection of enhancements from reference implementations"""
    
    @staticmethod
    def morphological_cleanup(anomaly_map, iterations=2):
        """
        From ADE: Morphological post-processing to reduce false positives
        """
        # Opening to remove small false positives
        cleaned = ndimage.binary_opening(anomaly_map > 0.5, iterations=iterations)
        # Closing to fill small gaps in anomalies
        cleaned = ndimage.binary_closing(cleaned, iterations=iterations)
        # Area filtering - remove very small components
        from scipy.ndimage import label
        labeled, num_features = label(cleaned)
        sizes = ndimage.sum(cleaned, labeled, range(num_features + 1))
        mask_sizes = sizes > 10  # Minimum 10 pixels
        mask_sizes[0] = 0  # Remove background
        cleaned = mask_sizes[labeled]
        return cleaned.astype(np.float32)
    
    @staticmethod
    def sliding_window_normalization(anomaly_scores, window_size=15):
        """
        From CRDBPSW: Local normalization for consistent detection
        """
        from scipy.ndimage import uniform_filter
        # Compute local statistics
        local_mean = uniform_filter(anomaly_scores, size=window_size, mode='reflect')
        local_var = uniform_filter(anomaly_scores**2, size=window_size, mode='reflect') - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        # Normalize
        normalized = (anomaly_scores - local_mean) / (local_std + 1e-8)
        return normalized
    
    @staticmethod
    def glrt_threshold(scores, background_region, alpha=0.01):
        """
        From GLRT: Statistical thresholding based on false alarm rate
        """
        # Estimate background distribution
        bg_scores = scores[background_region]
        mean_bg = bg_scores.mean()
        std_bg = bg_scores.std()
        
        # Compute adaptive threshold
        from scipy.stats import norm
        z_threshold = norm.ppf(1 - alpha)
        threshold = mean_bg + z_threshold * std_bg
        
        return scores > threshold
    
    @staticmethod
    def construct_background_dictionary(X, dict_size=50, sample_ratio=0.1):
        """
        From LRSR: Learn background dictionary for better sparse coding
        """
        H, W, B = X.shape
        X_flat = X.reshape(H*W, B)
        
        # Sample subset for efficiency
        n_samples = int(H * W * sample_ratio)
        indices = np.random.choice(H*W, n_samples, replace=False)
        X_sampled = X_flat[indices]
        
        # Learn dictionary
        dict_learner = DictionaryLearning(
            n_components=dict_size,
            alpha=0.1,
            max_iter=100,
            random_state=42,
            n_jobs=-1
        )
        dict_learner.fit(X_sampled)
        
        return dict_learner.components_
    
    @staticmethod
    def compute_band_weights(X, y=None, method='variance'):
        """
        From HyperAD: Compute importance weights for spectral bands
        """
        H, W, B = X.shape
        
        if method == 'variance':
            # Simple variance-based weighting
            X_flat = X.reshape(H*W, B)
            band_var = X_flat.var(axis=0)
            weights = band_var / band_var.sum()
            
        elif method == 'rf' and y is not None:
            # Random Forest importance
            X_flat = X.reshape(H*W, B)
            y_flat = y.reshape(H*W)
            
            # Only use labeled pixels
            mask = y_flat >= 0
            if mask.sum() > 100:  # Need enough samples
                rf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(X_flat[mask], y_flat[mask])
                weights = rf.feature_importances_
            else:
                weights = np.ones(B) / B
        else:
            weights = np.ones(B) / B
        
        return weights
    
    @staticmethod
    def rx_detector(X, window_size=15):
        """
        Classic RX detector as baseline comparison
        """
        H, W, B = X.shape
        X_flat = X.reshape(H*W, B)
        
        # Global statistics
        mean = X_flat.mean(axis=0)
        cov = np.cov(X_flat.T)
        cov_inv = np.linalg.pinv(cov)
        
        # Mahalanobis distance
        diff = X_flat - mean
        scores = np.sum(diff @ cov_inv * diff, axis=1)
        scores = scores.reshape(H, W)
        
        return scores


class EBTEnhanced(torch.nn.Module):
    """
    Enhanced EBT with reference-inspired components
    Add this to your main model for improved performance
    """
    
    def __init__(self, base_ebt_model):
        super().__init__()
        self.base_model = base_ebt_model
        self.enhancer = ReferenceEnhancements()
        
    def forward(self, X, S, A):
        # Base energy
        energy = self.base_model(X, S, A)
        
        # Add enhancement terms if needed
        return energy
    
    def enhanced_inference(self, X, S, num_iters=10):
        """
        Inference with reference enhancements
        """
        # Get base prediction
        A = self.base_model.iterative_refinement(X, S, num_iters)
        
        # Apply morphological cleanup
        A_cleaned = self.enhancer.morphological_cleanup(A)
        
        # Apply sliding window normalization
        A_normalized = self.enhancer.sliding_window_normalization(A_cleaned)
        
        return A_normalized


# Metrics from references
def compute_auc_metrics(scores, gt, num_thresholds=100):
    """
    Comprehensive AUC computation from reference implementations
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    scores_flat = scores.flatten()
    gt_flat = gt.flatten()
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(gt_flat, scores_flat)
    roc_auc = auc(fpr, tpr)
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(gt_flat, scores_flat)
    pr_auc = auc(recall, precision)
    
    # AUC at different FPR levels (partial AUC)
    target_fprs = [0.01, 0.05, 0.1]
    partial_aucs = {}
    for target_fpr in target_fprs:
        mask = fpr <= target_fpr
        if mask.sum() > 1:
            partial_aucs[f'auc_{int(target_fpr*100)}'] = auc(fpr[mask], tpr[mask])
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        **partial_aucs
    }
'''
        return integration_code


# Main execution
if __name__ == "__main__":
    extractor = ReferenceExtractor()
    
    # Get insights
    insights = extractor.analyze_references()
    
    print("=" * 60)
    print("REFERENCE REPOSITORY INSIGHTS")
    print("=" * 60)
    
    for repo_name, info in insights.items():
        print(f"\n📁 {repo_name.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Key concepts:")
        for concept in info['key_concepts']:
            print(f"      • {concept}")
    
    print("\n" + "=" * 60)
    print("GENERATING INTEGRATION MODULE...")
    print("=" * 60)
    
    # Generate integration code
    integration_code = extractor.generate_integration_module()
    
    # Save to file
    output_path = Path('/mnt/user-data/outputs/reference_enhancements.py')
    with open(output_path, 'w') as f:
        f.write(integration_code)
    
    print(f"✅ Integration module saved to: {output_path}")
    print("\nYou can now import and use these enhancements in your main code:")
    print("   from reference_enhancements import ReferenceEnhancements, EBTEnhanced")