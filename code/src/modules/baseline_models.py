import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MRNetBaseModel(nn.Module):
    """Backbone with max pooling across slices for MRI volume classification.
    Architecture: backbone features -> Global Avg Pool -> Max pool slices -> FC(feature_dim->1)
    Modeling after original MRNet architecture from the paper."""

    def __init__(self, backbone, feature_dim=256):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(feature_dim, 1)
    
    def forward(self, x):
        """Args: x (batch_size, num_slices, 3, H, W)
        Returns: logits (batch_size, 1)"""
        batch_size, num_slices = x.shape[0], x.shape[1]
        
        # Reshape to process all slices through backbone
        x = x.view(batch_size * num_slices, 3, x.shape[3], x.shape[4])
        features = self.backbone(x)
        
        # Global average pooling to reduce spatial dimensions
        features = F.adaptive_avg_pool2d(features, (1, 1))  
        
        # Flatten and reshape back to (batch_size, num_slices, 256)
        features = features.view(features.size(0), -1)
        features = features.view(batch_size, num_slices, -1)
        
        # Max pool across slices
        pooled, _ = torch.max(features, dim=1)
        logits = self.fc(pooled)
        return logits
    
    def forward_with_slice_tracking(self, x):
        """Extended forward pass for explainability.
        Returns: (logits, slice_indices) where slice_indices has shape (batch_size, feature_dim)"""
        batch_size, num_slices = x.shape[0], x.shape[1]
        
        x = x.view(batch_size * num_slices, 3, x.shape[3], x.shape[4])
        features = self.backbone(x)  
        
        # Global average pooling to reduce spatial dimensions
        features = F.adaptive_avg_pool2d(features, (1, 1))  
        
        features = features.view(features.size(0), -1)
        features = features.view(batch_size, num_slices, -1)
        
        # Max pool and track which slice contributed each feature
        pooled, slice_indices = torch.max(features, dim=1)
        logits = self.fc(pooled)
        return logits, slice_indices, pooled
    
    def freeze_backbone(self):
        """Freeze backbone parameters. Note: Not used for single-phase training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters. Note: Not used for single-phase training."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Returns parameters that require gradients (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]


def create_baseline_model():
    """Create baseline AlexNet model with ImageNet pretraining.
    Extracts convolutional features from AlexNet (designed for 1000 ImageNet classes)
    and replaces with custom FC(256->1) for binary classification.
    """
    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    # Extract only the convolutional feature layers
    backbone = alexnet.features
    model = MRNetBaseModel(backbone)
    return model
