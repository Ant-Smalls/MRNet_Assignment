import torch
import torch.nn as nn
import torchvision.models as models


class MRNetBaseModel(nn.Module):
    """ResNet18 backbone with max pooling across slices for MRI volume classification.
    Architecture: ResNet18 (no FC) -> Max pool slices -> FC(512->1)"""
    
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, 1)
    
    def forward(self, x):
        """Args: x (batch_size, num_slices, 3, H, W)
        Returns: logits (batch_size, 1)"""
        batch_size, num_slices = x.shape[0], x.shape[1]
        
        # Reshape to process all slices through backbone
        x = x.view(batch_size * num_slices, 3, x.shape[3], x.shape[4])
        features = self.backbone(x)
        
        # Flatten and reshape back to (batch_size, num_slices, 512)
        features = features.view(features.size(0), -1)
        features = features.view(batch_size, num_slices, -1)
        
        # Max pool across slices
        pooled, _ = torch.max(features, dim=1)
        logits = self.fc(pooled)
        return logits
    
    def forward_with_slice_tracking(self, x):
        """Extended forward pass for explainability.
        Returns: (logits, slice_indices) where slice_indices has shape (batch_size, 512)"""
        batch_size, num_slices = x.shape[0], x.shape[1]
        
        x = x.view(batch_size * num_slices, 3, x.shape[3], x.shape[4])
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = features.view(batch_size, num_slices, -1)
        
        # Max pool and track which slice contributed each feature
        pooled, slice_indices = torch.max(features, dim=1)
        logits = self.fc(pooled)
        return logits, slice_indices
    
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
    """Create baseline ResNet18 model with ImageNet pretraining.
    Removes final FC layer (designed for 1000 ImageNet classes) and replaces with 
    custom FC(512->1) for binary classification."""
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet18.children())[:-1])
    model = MRNetBaseModel(backbone)
    return model
