# Beyond Generation: Energy-Based Transformers for Hyperspectral Anomaly Detection

## Project Overview

We're implementing a novel approach to hyperspectral and thermal anomaly detection using Energy-Based Transformers. This is for the **AI Grand Challenge PS-11 competition** with a submission deadline of **October 31, 2025**.

### What Makes This Novel?

Instead of training models to **generate** anomaly maps directly (like all other approaches), we treat anomaly detection as **verification**. Our model learns an energy function that evaluates whether a candidate anomaly map is compatible with the observed data, then refines predictions through iterative optimization. This is called **System 2 thinking** in AI - deliberate reasoning rather than reflexive pattern matching.

### The Two-Stage Architecture

**Stage 1: Low-Rank Sparse Representation (LRSR)**
- Classical optimization using ADMM algorithm
- Decomposes hyperspectral data into low-rank background + sparse anomalies
- Fast O(n log n) processing for bulk data filtering
- Runs in CPU, no GPU needed

**Stage 2: Energy-Based Transformer (EBT)**
- Deep learning refinement through iterative energy minimization
- Verifies and refines the sparse anomalies from Stage 1
- Implements System 2 thinking through gradient descent on energy landscape
- GPU-accelerated for complex reasoning

**Multimodal Fusion:**
- Joint energy function combining hyperspectral (HSI) and thermal infrared (TIR)
- Natural integration of complementary evidence through unified optimization
- First-of-its-kind in remote sensing

---

## Competition Requirements

### Datasets We Need to Handle

**Training Data (Download These):**
- Hyperspectral: ABU (Airport/Beach/Urban), Salinas, San Diego, HYDICE Urban
- Thermal: Landsat-8/9, FLIR datasets, various thermal imagery

**Test Data (Will be provided):**
- PRISMA/EnMAP hyperspectral (30km × 30km scenes)
- Landsat-8/9 thermal bands
- Mock competition data on Oct 30, 2025
- Final evaluation at IIT Delhi

### Evaluation Metrics
- **Primary:** PR-AUC (Precision-Recall Area Under Curve)
- **Secondary:** F1-Score, ROC-AUC
- **Constraints:** 2-hour processing time limit on A100 GPU

### Submission Requirements
1. Model hash value (SHA-256)
2. Anomaly detection results in GeoTIFF and PNG formats
3. Excel report with accuracy metrics and hardware specs
4. All files named with team identifier

---

## Project Structure

```
hyperspectral-ebt/
├── data/
│   ├── raw/              # Original downloaded datasets
│   ├── processed/        # Preprocessed tensors
│   └── cache/           # LRSR cached outputs
├── src/
│   ├── config.py        # All hyperparameters (SHARED)
│   ├── stage1_lrsr.py   # LRSR implementation (PERSON 1)
│   ├── stage2_ebt.py    # EBT architecture (PERSON 3)
│   ├── thermal_model.py # Thermal CNN (PERSON 3)
│   ├── data_loader.py   # HSI datasets (PERSON 2)
│   ├── thermal_data_loader.py  # Thermal datasets (PERSON 2)
│   ├── preprocessing.py # Data preprocessing (PERSON 2)
│   ├── train.py         # HSI training loop (PERSON 1)
│   ├── train_thermal.py # Thermal training (PERSON 1)
│   ├── inference.py     # Inference pipeline (PERSON 3)
│   ├── metrics.py       # Evaluation metrics (PERSON 2)
│   ├── multimodal_fusion.py  # HSI-TIR fusion (PERSON 1)
│   └── utils.py         # Helper functions (SHARED)
├── experiments/logs/
├── models/checkpoints/
├── results/
│   ├── visualizations/
│   └── submissions/
├── requirements.txt
├── README.md
└── run_training.py      # Main script (PERSON 1)
```

---

## Setup Instructions

### 1. Environment Setup

```bash
# Clone/Navigate to project
cd hyperspectral-ebt

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install torch torchvision numpy scipy scikit-learn spectral rasterio opencv-python matplotlib seaborn pandas tqdm pyyaml wandb h5py
```

**Note:** We removed GDAL from requirements because it's a pain to install on Windows. Rasterio handles all geospatial operations we need.

### 2. Verify Installation

```python
import torch
import numpy as np
import rasterio
from spectral import *

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"NumPy: {np.__version__}")
```

### 3. Project Configuration

All hyperparameters are in `src/config.py`. Key parameters:

```python
# LRSR parameters
lrsr_lambda_l = 0.01    # Low-rank weight
lrsr_lambda_s = 0.1     # Sparse weight

# EBT parameters
embed_dim = 256
num_heads = 8
num_layers = 6

# Training
batch_size = 16
learning_rate = 1e-4
num_epochs = 50

# Inference
num_refinement_iters = 10
```

---

## Work Division

### **Person 1 (Team Lead) - 40% of Work**

**Core Responsibilities:**
1. Set up complete project infrastructure
2. Implement LRSR (Stage 1) using ADMM algorithm
3. Build training loops for both HSI and thermal models
4. Implement multimodal fusion energy function
5. Integrate all components into unified pipeline
6. Handle competition submission and hash generation

**Key Files You Own:**
- `src/stage1_lrsr.py` - LRSR decomposition
- `src/train.py` - HSI training loop
- `src/train_thermal.py` - Thermal training loop
- `src/multimodal_fusion.py` - Joint energy function
- `run_training.py` - Main execution script

**Your Starting Point:**
The LRSR code is already written in `src/stage1_lrsr.py`. Your first task:
1. Test it on a small hyperspectral image
2. Set up the training infrastructure
3. Create the main execution pipeline

---

### **Person 2 (Data & Metrics) - 30% of Work**

**Core Responsibilities:**
1. Download ALL training datasets from competition links
2. Implement data loaders for HSI and thermal
3. Handle preprocessing (normalization, band selection, augmentation)
4. Implement all evaluation metrics (F1, ROC-AUC, PR-AUC)
5. Create visualization tools
6. Generate submission materials (GeoTIFF, PNG, Excel reports)

**Key Files You Own:**
- `src/data_loader.py` - Hyperspectral dataset class
- `src/thermal_data_loader.py` - Thermal dataset class
- `src/preprocessing.py` - All preprocessing functions
- `src/metrics.py` - Evaluation metrics
- `src/utils.py` - Visualization and export utilities

**Your Starting Tasks:**

#### Task 1: Download Datasets (Day 1)
Download from these links provided in competition document:
- ABU datasets: https://zephyrhours.github.io/sources.html
- Additional HSI: https://paperswithcode.com/datasets
- Thermal: Landsat-8/9 data from IEEE DataPort links

Organize in this structure:
```
data/raw/
├── hsi/
│   ├── abu_airport/
│   ├── abu_beach/
│   ├── salinas/
│   └── ...
└── thermal/
    ├── landsat/
    └── flir/
```

#### Task 2: Implement Data Loaders (Day 1-2)

**For `src/data_loader.py`:**
```python
def _load_samples(self):
    """Scan data_dir and create list of samples"""
    # TODO: Implement this function
    # 1. Scan data/raw/hsi/ directory
    # 2. For each dataset, find image and ground truth pairs
    # 3. Return list of dicts with paths
    samples = []
    # Your code here
    return samples
```

**For `src/thermal_data_loader.py`:**
```python
def _read_thermal_image(self, path):
    """Read and convert Landsat thermal to temperature"""
    # TODO: Implement Landsat thermal conversion
    # 1. Read thermal band DN values
    # 2. Convert to radiance
    # 3. Convert to brightness temperature
    pass
```

#### Task 3: Implement Metrics (Day 2-3)

Complete `src/metrics.py` with all required functions:
- Precision, Recall, F1-Score
- ROC-AUC computation
- PR-AUC computation
- Confusion matrix generation

#### Task 4: Visualization Tools (Day 3-4)

Create functions to:
- Display RGB composites from hyperspectral
- Overlay predictions on input images
- Generate side-by-side comparisons
- Export to PNG and GeoTIFF formats

---

### **Person 3 (Neural Networks & Inference) - 30% of Work**

**Core Responsibilities:**
1. Complete the EBT architecture for hyperspectral
2. Build thermal anomaly detection CNN (U-Net style)
3. Implement inference pipelines with iterative refinement
4. Optimize for speed (tile-based processing, mixed precision)
5. Handle both single-modal and multimodal inference

**Key Files You Own:**
- `src/stage2_ebt.py` - Energy-Based Transformer
- `src/thermal_model.py` - Thermal CNN architecture
- `src/inference.py` - Complete inference pipeline

**Your Starting Tasks:**

#### Task 1: Understand EBT Architecture (Day 1)

Study `src/stage2_ebt.py`. The architecture has:
- **Spectral Encoder**: Processes hyperspectral signatures
- **Spatial Encoder**: Extracts spatial patterns
- **Transformer**: Reasons about spatial-spectral relationships
- **Energy Head**: Outputs scalar energy value

Your job: Make sure the forward pass works correctly.

Test it:
```python
from src.stage2_ebt import EnergyBasedTransformer

model = EnergyBasedTransformer()
X = torch.randn(1, 256, 256, 200)  # Fake HSI data
S = torch.randn(1, 256, 256, 200)  # Fake sparse component
A = torch.randn(1, 256, 256)       # Fake anomaly map

energy = model(X, S, A)
print(f"Energy: {energy.item()}")  # Should output a scalar
```

#### Task 2: Build Thermal CNN (Day 1-2)

Complete `src/thermal_model.py`. You need:
- U-Net encoder-decoder architecture
- Forward pass for both 'direct' and 'energy' modes
- Thermal deviation energy computation

#### Task 3: Implement Iterative Refinement (Day 2-3)

In `src/inference.py`, complete the `iterative_refinement` method:

```python
def iterative_refinement(self, X, S, num_iters=10):
    """
    Refine anomaly map through gradient descent on energy
    
    Key steps:
    1. Initialize A from thresholded sparse component
    2. For each iteration:
       - Compute energy E(X, S, A)
       - Compute gradient dE/dA
       - Update A = A - alpha * gradient + noise
       - Clamp A to [0, 1]
    3. Return final A
    """
    # Your implementation here
```

#### Task 4: Tile-Based Processing (Day 3-4)

Implement `process_large_image()` for handling 30km scenes:
- Divide image into 256×256 tiles with 32px overlap
- Process each tile through inference
- Aggregate with weighted averaging in overlap regions

#### Task 5: Speed Optimization (Day 4-5)

Optimize inference to meet 2-hour deadline:
- Use `torch.no_grad()` where appropriate
- Implement mixed precision inference
- Batch process multiple tiles
- Profile and identify bottlenecks

---

## Development Timeline (13 Days)

### Days 1-2: Foundation
- **Person 1:** Set up project, test LRSR
- **Person 2:** Download datasets, start data loaders
- **Person 3:** Understand architectures, test forward passes

### Days 3-5: Core Implementation
- **Person 1:** Build training loops, test on small data
- **Person 2:** Complete data loaders and metrics
- **Person 3:** Implement iterative refinement

### Days 6-8: Integration
- **Person 1:** Connect all components, first end-to-end training
- **Person 2:** Generate visualizations, test metrics
- **Person 3:** Optimize inference speed

### Days 9-11: Training & Tuning
- **All:** Train models on full datasets
- **All:** Hyperparameter tuning
- **All:** Validate on benchmark data

### Days 12-13: Competition Prep
- **Person 1:** Multimodal fusion, final integration
- **Person 2:** Generate submission materials
- **Person 3:** Final speed optimizations
- **All:** Test on mock competition data (Oct 30)

---

## Key Mathematical Concepts

### LRSR Decomposition
Given hyperspectral matrix X, we solve:
```
minimize: ||L||* + λ_L||L||²_F + λ_S||S||_1
subject to: X = L + S
```
- `||L||*` = nuclear norm (sum of singular values) → encourages low rank
- `||S||_1` = L1 norm (sum of absolute values) → encourages sparsity
- L captures correlated background
- S captures rare anomalies

### Energy-Based Learning
Instead of learning `f: X → A` (generation), we learn `E: (X, A) → R` (verification).

Training minimizes:
```
L = E(X, A_real) - E(X, A_fake)
```
- Low energy on real anomaly maps
- High energy on fake anomaly maps

Inference finds:
```
A* = argmin_A E(X, A)
```
through gradient descent.

### Iterative Refinement (System 2 Thinking)
```
A_{t+1} = A_t - α_t ∇_A E(X, A_t) + noise
```
- Starts with coarse guess
- Progressively refines through gradient steps
- Adds noise for exploration (Langevin dynamics)
- Converges to low-energy configuration

---

## Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size or tile size in `config.py`

### Issue: LRSR Too Slow
**Solution:** Reduce `lrsr_max_iters` or increase `lrsr_tol`

### Issue: Training Not Converging
**Solution:** Check negative sample generation, adjust learning rate

### Issue: Poor Generalization
**Solution:** Add more data augmentation, increase model regularization

---

## Testing Your Code

### Test LRSR
```python
from src.stage1_lrsr import LRSR
import numpy as np

# Create fake data
X = np.random.randn(100, 100, 50)  # Small hyperspectral image
lrsr = LRSR()
L, S = lrsr(X)

print(f"Background rank: {np.linalg.matrix_rank(L.reshape(-1, 50).T)}")
print(f"Anomaly sparsity: {np.sum(S != 0) / S.size}")
```

### Test EBT
```python
from src.stage2_ebt import EnergyBasedTransformer
import torch

model = EnergyBasedTransformer()
X = torch.randn(2, 256, 256, 200)
S = torch.randn(2, 256, 256, 200)
A = torch.randn(2, 256, 256)

energy = model(X, S, A)
print(f"Energy shape: {energy.shape}")  # Should be scalar per batch
```

### Test Data Loading
```python
from src.data_loader import get_dataloaders

train_loader, val_loader = get_dataloaders(batch_size=4)
batch = next(iter(train_loader))

print(f"Image shape: {batch['image'].shape}")
print(f"GT shape: {batch['gt'].shape}")
```

---

## Communication Protocol

### Daily Standups (15 min)
- What did you complete yesterday?
- What are you working on today?
- Any blockers?

### Code Integration
- Test your code before pushing
- Use descriptive commit messages
- Update this README with any changes

### Asking for Help
- Post in team chat with specific error messages
- Include minimal code to reproduce issue
- Tag relevant person

---

## Resources

### Competition Links
- PS-11 Document: [Provided in project]
- Hyperspectral datasets: Listed in competition doc
- Thermal datasets: IEEE DataPort

### Technical References
- LRSR: "Robust PCA via Alternating Directions Method"
- Energy-Based Models: LeCun et al., "A Tutorial on Energy-Based Learning"
- System 2 Thinking: "Thinking, Fast and Slow" by Kahneman

### Code References
- PyTorch Docs: https://pytorch.org/docs/
- Rasterio Docs: https://rasterio.readthedocs.io/
- SPy (Spectral): http://www.spectralpython.net/

---

## Final Notes

**This is your roadmap to pole position.** The theory is novel, the architecture is sound, and the timeline is tight but achievable. Focus on:

1. **Parallel work**: Don't wait for others, work on your components independently
2. **Testing**: Test each component before integration
3. **Communication**: Brief daily sync-ups prevent blocking issues
4. **Pragmatism**: Start simple, add complexity only if time permits

The competition deadline is **October 31, 2025 at 23:59**. We need working models by October 28 to allow time for testing and submission prep.

Let's build something remarkable. 🚀