"""
Role 2: Baseline Models

ResNet18-based baseline model with max pooling slice aggregation.
Provides shared MRNetBaseModel class for use by Role 3.
"""

import torch
import torch.nn as nn
from torchvision import models


class MRNetBaseModel(nn.Module):
    """
    Base model class for MRNet with shared components.
    
    Architecture:
        - Backbone (feature extractor): ResNet18 without final FC
        - Process all slices: (S, 3, H, W) -> (S, 512) features
        - Max pool across slices: (S, 512) -> (512)
        - Binary classifier: (512) -> (1) logit
    
    Args:
        backbone (nn.Module): Pretrained feature extractor (e.g., ResNet18)
    """
    
    def __init__(self, backbone):
        super(MRNetBaseModel, self).__init__()
        
        self.backbone = backbone  # Feature extractor (ResNet without final FC)
        
        # TODO: Define max pooling layer (across slice dimension)
        self.global_pool = None
        
        # TODO: Define binary classification head (512 -> 1)
        self.fc = None
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input of shape (batch_size, num_slices, 3, H, W)
                             For batch_size=1: (1, S, 3, H, W)
        
        Returns:
            torch.Tensor: Logit of shape (batch_size, 1)
        
        TODO:
        - Reshape to process all slices: (B, S, 3, H, W) -> (B*S, 3, H, W)
        - Pass through backbone: (B*S, 3, H, W) -> (B*S, 512)
        - Reshape back: (B*S, 512) -> (B, S, 512)
        - Max pool across slices: (B, S, 512) -> (B, 512)
        - Binary classifier: (B, 512) -> (B, 1)
        """
        pass
    
    def freeze_backbone(self):
        """
        Freeze all backbone parameters (for Phase 1 training).
        
        TODO: Set requires_grad=False for all backbone parameters
        """
        pass
    
    def unfreeze_backbone(self, num_layers=None):
        """
        Unfreeze backbone parameters (for Phase 2 fine-tuning).
        
        Args:
            num_layers (int, optional): If specified, only unfreeze last N blocks.
                                       If None, unfreeze all.
        
        TODO: Set requires_grad=True for backbone parameters
        """
        pass
    
    def get_trainable_params(self):
        """
        Get list of trainable parameters for optimizer.
        
        Returns:
            list: Parameters where requires_grad=True
        """
        return [p for p in self.parameters() if p.requires_grad]


def create_baseline_model():
    """
    Factory function to create baseline ResNet18 model.
    
    Returns:
        MRNetBaseModel: Baseline model with ImageNet pretrained ResNet18
        
    TODO:
    - Load pretrained ResNet18 from torchvision
    - Remove final FC layer (keep only feature extractor)
    - Create MRNetBaseModel with this backbone
    - Return model
    """
    # TODO: Load ResNet18
    # backbone = models.resnet18(pretrained=True)
    # backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final FC
    # model = MRNetBaseModel(backbone)
    # return model
    pass


if __name__ == '__main__':
    # TODO: Test model creation and forward pass
    # model = create_baseline_model()
    # print(f"Model created successfully")
    # 
    # # Test freeze/unfreeze
    # model.freeze_backbone()
    # print(f"Trainable params (frozen): {len(model.get_trainable_params())}")
    # 
    # model.unfreeze_backbone()
    # print(f"Trainable params (unfrozen): {len(model.get_trainable_params())}")
    pass
