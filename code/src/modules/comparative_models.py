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
            - 'resnet50'
            - 'radimagenet_resnet50'

    Returns:
        MRNetBaseModel
    """

    if architecture == 'resnet50':

        resnet50 = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        backbone = nn.Sequential(
            *list(resnet50.children())[:-1]
        )

        model = MRNetBaseModel(backbone)

        # ResNet50 outputs 2048 features
        model.fc = nn.Linear(2048, 1)

        return model

    elif architecture == 'radimagenet_resnet50':

        backbone = load_radimagenet_model(architecture)

        model = MRNetBaseModel(backbone)

        # ResNet50 outputs 2048 features
        model.fc = nn.Linear(2048, 1)

        return model

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

    model = create_comparative_model(
        'radimagenet_resnet50'
    )

    print(model)
    print("\nModel created successfully!")