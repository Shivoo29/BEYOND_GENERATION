import torch
from src.stage1_lrsr import LRSR
from src.stage2_ebt import EnergyBasedTransformer
from src.thermal_model import ThermalAnomalyDetector
from src.multimodal_fusion import MultimodalEnergyFunction, MultimodalInference
from src.config import config

def train_hyperspectral():
    """Train HSI anomaly detection model"""
    from src.data_loader import get_dataloaders
    from src.train import Trainer
    
    model = EnergyBasedTransformer()
    train_loader, val_loader = get_dataloaders()
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train()

def train_thermal():
    """Train thermal anomaly detection model"""
    from src.thermal_data_loader import get_thermal_dataloaders
    from src.train_thermal import ThermalTrainer
    
    model = ThermalAnomalyDetector(mode='direct')
    train_loader, val_loader = get_thermal_dataloaders()
    trainer = ThermalTrainer(model, train_loader, val_loader)
    trainer.train()

def train_multimodal():
    """Train joint HSI-TIR fusion model"""
    # Load pre-trained components
    hsi_model = EnergyBasedTransformer()
    tir_model = ThermalAnomalyDetector(mode='energy')
    
    # Initialize fusion model with pre-trained components
    fusion_model = MultimodalEnergyFunction()
    fusion_model.hsi_energy_fn = hsi_model
    fusion_model.tir_energy_fn = tir_model
    
    # Train only the cross-modal components
    # TODO: Implement fusion training loop

def inference_competition_data(hsi_path, tir_path, output_path):
    """Run inference on competition test data"""
    # Load models
    fusion_model = MultimodalEnergyFunction()
    fusion_model.load_state_dict(torch.load('models/fusion_best.pt'))
    
    # Initialize inference
    inference = MultimodalInference(fusion_model)
    
    # Load data
    X_hsi = load_hyperspectral(hsi_path)
    T_tir = load_thermal(tir_path)
    
    # Run joint inference
    anomaly_map, energies = inference.joint_refinement(X_hsi, None, T_tir)
    
    # Save results
    save_results(anomaly_map, output_path)

if __name__ == '__main__':
    # Training pipeline
    print("Training hyperspectral model...")
    train_hyperspectral()
    
    print("Training thermal model...")
    train_thermal()
    
    print("Training multimodal fusion...")
    train_multimodal()
    
    # Competition inference
    print("Running competition inference...")
    inference_competition_data(
        'data/competition/prisma_scene.tif',
        'data/competition/landsat_thermal.tif',
        'results/competition_output.tif'
    )