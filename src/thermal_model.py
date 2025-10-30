import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config

class ThermalEncoder(nn.Module):
    """
    Encoder for thermal imagery following U-Net style architecture
    This is simpler than the hyperspectral encoder because thermal
    has fewer channels but still needs spatial reasoning
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Encoder path - progressively downsample while increasing channels
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
    def _conv_block(self, in_ch, out_ch):
        """Standard convolutional block with batch norm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass returns multi-scale features"""
        # x: (B, 1, H, W) for single thermal band
        e1 = self.enc1(x)  # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8, W/8)
        
        return [e1, e2, e3, e4]

class ThermalDecoder(nn.Module):
    """
    Decoder that reconstructs spatial resolution with skip connections
    This allows the network to preserve fine spatial details while
    also incorporating high-level semantic understanding
    """
    def __init__(self, out_channels=1):
        super().__init__()
        
        # Decoder path with upsampling
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(512, 256)  # 512 because of skip connection
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        
        # Final output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, enc_features):
        """Decode with skip connections from encoder"""
        e1, e2, e3, e4 = enc_features
        
        # Progressively upsample and concatenate with encoder features
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)  # Skip connection
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)
        
        out = self.out(d3)
        return out

class ThermalAnomalyDetector(nn.Module):
    """
    Complete thermal anomaly detection model
    Uses encoder-decoder architecture to produce anomaly maps
    Can also be used in energy-based framework
    """
    def __init__(self, mode='direct'):
        super().__init__()
        self.mode = mode  # 'direct' for standard prediction, 'energy' for EBT
        
        self.encoder = ThermalEncoder(in_channels=1)
        self.decoder = ThermalDecoder(out_channels=1)
        
        # For energy-based mode
        if mode == 'energy':
            self.energy_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
    
    def forward(self, thermal_img, anomaly_map=None):
        """
        Forward pass with two modes:
        - Direct mode: predict anomaly map from thermal image
        - Energy mode: evaluate energy of (thermal_img, anomaly_map) pair
        """
        enc_features = self.encoder(thermal_img)
        
        if self.mode == 'direct':
            # Standard prediction mode
            pred = self.decoder(enc_features)
            return torch.sigmoid(pred)
        
        elif self.mode == 'energy':
            # Energy-based mode for verification
            assert anomaly_map is not None, "Energy mode requires anomaly_map"
            
            # Combine thermal features with anomaly hypothesis
            anomaly_features = F.interpolate(
                anomaly_map.unsqueeze(1), 
                size=enc_features[-1].shape[2:],
                mode='bilinear'
            )
            combined = torch.cat([enc_features[-1], anomaly_features], dim=1)
            
            # Compute energy score
            energy = self.energy_head(combined)
            
            # Add thermal-specific energy terms
            thermal_deviation = self._thermal_deviation_energy(thermal_img, anomaly_map)
            
            return energy + thermal_deviation
    
    def _thermal_deviation_energy(self, thermal_img, anomaly_map):
        """
        Energy term that measures whether detected anomalies correspond
        to thermal deviations from local background temperature
        """
        # Compute local background statistics using average pooling
        local_mean = F.avg_pool2d(thermal_img, kernel_size=15, stride=1, padding=7)
        
        # Deviation from local background
        deviation = torch.abs(thermal_img - local_mean)
        
        # Energy is low when anomalies align with high deviations
        energy = -(anomaly_map.unsqueeze(1) * deviation).mean()
        
        return energy