"""
Role 3: Comparative Models

Alternative architectures for comparison against baseline.
Reuses MRNetBaseModel from Role 2, swaps out backbone only.
"""

import torch
import torch.nn as nn
from torchvision import models
from baseline_models import MRNetBaseModel


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
    """

    if architecture == 'resnet50':

        resnet50 = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        backbone = nn.Sequential(
            *list(resnet50.children())[:-1]
        )

        model = MRNetBaseModel(backbone)

        # ResNet50 outputs 2048 features instead of 512
        model.fc = nn.Linear(2048, 1)

        return model

    elif architecture.startswith('radimagenet'):
        backbone = load_radimagenet_model(architecture)

        model = MRNetBaseModel(backbone)

        if architecture == 'radimagenet_resnet50':
            model.fc = nn.Linear(2048, 1)

        return model

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def load_radimagenet_model(architecture):
    """
    Placeholder for RadImageNet implementation.
    """

    raise NotImplementedError(
        "RadImageNet support not implemented yet."
    )


if __name__ == '__main__':

    model = create_comparative_model('resnet50')

    print(model)
    print("\nModel created successfully!")