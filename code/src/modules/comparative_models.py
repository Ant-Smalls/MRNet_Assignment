"""
Role 3: Comparative Models

Alternative architectures for comparison against baseline.
Reuses MRNetBaseModel from Role 2, swaps out backbone only.
"""

import torch
import torch.nn as nn
from torchvision import models
from .baseline_models import MRNetBaseModel


def create_comparative_model(architecture='resnet50'):
    """
    Factory function to create comparative models.
    
    Args:
        architecture (str): One of:
            - 'resnet50': Deeper ResNet with ImageNet weights
            - 'radimagenet_resnet18': ResNet18 pretrained on RadImageNet
            - 'radimagenet_resnet50': ResNet50 pretrained on RadImageNet
    
    Returns:
        MRNetBaseModel: Model with specified architecture
        
    TODO:
    - Load appropriate backbone based on architecture parameter
    - For ResNet50: Use torchvision.models.resnet50(pretrained=True)
    - For RadImageNet: Load custom pretrained weights
    - Remove final FC layer from backbone
    - Create MRNetBaseModel with this backbone
    - Return model
    """
    
    if architecture == 'resnet50':
        # TODO: Load ResNet50
        # backbone = models.resnet50(pretrained=True)
        # backbone = nn.Sequential(*list(backbone.children())[:-1])
        # model = MRNetBaseModel(backbone)
        # return model
        pass
        
    elif architecture.startswith('radimagenet'):
        # TODO: Load RadImageNet pretrained weights
        # This requires downloading RadImageNet weights separately
        # backbone = load_radimagenet_model(architecture)
        # model = MRNetBaseModel(backbone)
        # return model
        pass
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_radimagenet_model(architecture):
    """
    Load RadImageNet pretrained ResNet50 backbone.
    """

    if architecture != "radimagenet_resnet50":
        raise NotImplementedError(
            f"{architecture} not supported."
        )

    resnet50 = models.resnet50(weights=None)

    backbone = nn.Sequential(
        *list(resnet50.children())[:-1]
    )

    checkpoint = torch.load(
        "models/ResNet50.pt",
        map_location="cpu"
    )

    # Remove "backbone." prefix from checkpoint keys
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if k.startswith("backbone."):
            new_checkpoint[k.replace("backbone.", "", 1)] = v

    backbone.load_state_dict(new_checkpoint)

    return backbone


if __name__ == '__main__':
    # TODO: Test model creation for different architectures
    # model_resnet50 = create_comparative_model('resnet50')
    # print(f"ResNet50 model created")
    # 
    # # Test that it has same interface as baseline
    # model_resnet50.freeze_backbone()
    # print(f"Freeze/unfreeze API works")
    pass
