# 🎯 REMAINING WORK - ACTION PLAN

## Current Status ✅
- ✅ Core EBT architecture implemented
- ✅ LRSR Stage 1 working
- ✅ Training/inference pipelines ready
- ✅ HSI data available (abu, salinas, hydice, pavia)
- ❌ Thermal data missing
- ❌ Models not trained yet
- ❌ Reference code not integrated

## 📋 IMMEDIATE TASKS (Next 48 Hours)

### Person 1 (Team Lead) - START NOW
1. **Test the updated data loader**:
```bash
cd hyperspectral-ebt
python -c "from data_loader_updated import get_dataloaders; get_dataloaders(test_mode=True)"
```

2. **Start training immediately** (this takes hours!):
```bash
# Use the updated data loader
cp /mnt/user-data/outputs/data_loader_updated.py src/data_loader.py

# Start training with your data
python run_training.py \
    --train_hsi \
    --num_epochs 30 \
    --batch_size 4 \
    --data_dir ./data \
    --mixed_precision
```

3. **Monitor training** and adjust hyperparameters if loss isn't decreasing

### Person 2 (Data Expert) - URGENT
1. **Download Thermal Data TODAY**:
   - Go to: https://earthexplorer.usgs.gov/
   - Download Landsat 8/9 Collection 2 Level-2
   - Get thermal bands (B10, B11) for same regions as HSI data
   - Alternative: https://glovis.usgs.gov/

2. **Organize thermal data**:
```bash
mkdir -p data/raw/thermal/landsat
# Place downloaded .TIF files there
```

3. **Test data loading**:
```python
from thermal_data_loader import get_thermal_dataloaders
train_loader, val_loader = get_thermal_dataloaders()
```

### Person 3 (Optimization) - CRITICAL
1. **Profile inference speed**:
```python
import time
from run_competition_inference import CompetitionInference

inference = CompetitionInference(args)
start = time.time()
result = inference.process_large_image(test_image)
print(f"Time: {time.time() - start}s")
```

2. **Optimize bottlenecks**:
   - Enable mixed precision
   - Increase batch size for tile processing
   - Reduce refinement iterations if too slow

## 🔧 Integration Tasks

### 1. Use the Updated Data Loader
```bash
# Replace the old data loader
cp /mnt/user-data/outputs/data_loader_updated.py src/data_loader.py
```

### 2. Add Reference Enhancements
```bash
# Copy enhancement module
cp /mnt/user-data/outputs/reference_enhancements.py src/

# In your training script, add:
from reference_enhancements import ReferenceEnhancements, compute_auc_metrics

# Use enhanced metrics
metrics = compute_auc_metrics(predictions, ground_truth)
```

### 3. Quick Test with Your Data
```python
# Test script to verify everything works
import sys
sys.path.append('.')

from src.data_loader_updated import get_dataloaders
from src.stage1_lrsr import LRSR
from src.stage2_ebt import EnergyBasedTransformer
import torch

# Test data loading
print("Testing data loader...")
train_loader, val_loader = get_dataloaders(batch_size=2, test_mode=True)
print(f"✅ Data loader works! Found {len(train_loader)} batches")

# Test LRSR
print("\nTesting LRSR...")
lrsr = LRSR()
import numpy as np
test_data = np.random.randn(100, 100, 50)
L, S = lrsr(test_data)
print(f"✅ LRSR works! Sparse ratio: {(S!=0).sum()/S.size:.2%}")

# Test EBT
print("\nTesting EBT model...")
model = EnergyBasedTransformer()
X = torch.randn(1, 256, 256, 200)
S = torch.randn(1, 256, 256, 200)
A = torch.randn(1, 256, 256)
energy = model(X, S, A)
print(f"✅ EBT works! Energy: {energy.item():.4f}")

print("\n🎉 All components working!")
```

## 📊 Hyperparameter Recommendations

Based on the reference implementations, adjust these in `config.py`:

```python
# LRSR (from reference/lrsr)
lrsr_lambda_l = 0.01  # Lower = more low-rank
lrsr_lambda_s = 0.05  # Lower = more sparse (more anomalies)

# EBT Training
learning_rate = 5e-4  # Start higher
batch_size = 4  # Smaller for memory
margin = 1.0  # Contrastive margin

# Inference
num_refinement_iters = 15  # More iterations = better quality
langevin_noise = 0.005  # Less noise = more stable
```

## 🏃 Daily Schedule Until Oct 31

### Oct 23-24 (Days 1-2)
- ✅ Get data loader working with your data
- ✅ Start HSI model training (let it run overnight)
- ✅ Download thermal data

### Oct 25-26 (Days 3-4)  
- Continue training, monitor metrics
- Test inference on validation data
- Implement reference enhancements

### Oct 27-28 (Days 5-6)
- Hyperparameter tuning based on validation
- Speed optimization for 2-hour limit
- Create preliminary submission package

### Oct 29 (Day 7)
- Final model training
- Ensemble multiple checkpoints if time permits
- Prepare submission materials

### Oct 30 (Day 8) - MOCK TEST
- Test on mock competition data
- Fix any issues found
- Final optimization

### Oct 31 (Day 9) - SUBMISSION DAY
- Morning: Final inference run
- Afternoon: Generate submission package
- Evening: Submit before 23:59!

## 🚨 Critical Success Factors

1. **START TRAINING NOW** - It takes hours/days!
2. **Use lower batch size** (4 or 2) to avoid OOM
3. **Save checkpoints frequently** - Every 5 epochs
4. **Test on small data first** - Use test_mode=True
5. **Monitor GPU memory** - Use nvidia-smi

## 💻 Useful Commands

### Monitor Training
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f experiments/logs/training.log

# TensorBoard (if using)
tensorboard --logdir experiments/logs
```

### Quick Validation
```python
# Load checkpoint and evaluate
checkpoint = torch.load('models/checkpoints/hsi_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
metrics = validate_model(model, val_loader)
print(f"Validation PR-AUC: {metrics['pr_auc']:.4f}")
```

## 🔥 Emergency Fixes

### If OOM Error:
```python
# In config.py
batch_size = 2  # Reduce
tile_size = 128  # Reduce
mixed_precision = True  # Enable
```

### If Training Not Converging:
```python
# Adjust learning rate
learning_rate = 1e-3  # Try higher
# Or use learning rate finder
```

### If Data Not Loading:
```python
# Debug data loader
dataset = HyperspectralDataset(data_dir='./data', split='train')
print(f"Found {len(dataset)} samples")
# Check data paths
```

## 📞 Help Resources

1. **Implementation questions**: Review the IMPLEMENTATION_README.md
2. **Theory questions**: Check the original README.md on 4th commit on github
3. **Data format issues**: Use scipy.io.whosmat('file.mat') to inspect
4. **GPU issues**: Try CPU training first with --device cpu

## ✅ Final Checklist Before Submission

- [ ] Model trained for at least 30 epochs
- [ ] Validation PR-AUC > 0.7
- [ ] Inference time < 2 hours on test image
- [ ] Submission package includes:
  - [ ] GeoTIFF anomaly map
  - [ ] PNG visualization
  - [ ] Excel report with metrics
  - [ ] Model hash (SHA-256)
  - [ ] Team identifier in filenames

## 🎯 GO GET THAT POLE POSITION! 

You have everything you need. The implementation is solid, the approach is novel, and you have 9 days to execute. Focus on:

1. **Get training started NOW**
2. **Download thermal data TODAY**  
3. **Test everything works with YOUR data**
4. **Iterate and improve daily**

The clock is ticking - let's make it happen! 🚀