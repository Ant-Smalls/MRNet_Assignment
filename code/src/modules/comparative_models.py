"""
Role 3: Comparative Models

Two comparison backbones against the AlexNet baseline:
  - resnet50:    ResNet50 pretrained on ImageNet (deeper architecture comparison)
  - xrv_dense:   DenseNet121 pretrained on chest X-rays via TorchXRayVision
                 (medical-domain pretraining comparison, NOT trained on MRNet/Stanford MRI)

Both reuse MRNetBaseModel from Role 2, swapping only the backbone.
The FC head dimension is set to match each backbone's output channels.

Backbone output channels:
  AlexNet features  -> 256
  ResNet50 layer4   -> 2048
  DenseNet121 feats -> 1024
"""

import torch
import torch.nn as nn
from torchvision import models
from .baseline_models import MRNetBaseModel


# ---------------------------------------------------------------------------
# ResNet50 (ImageNet) — deeper architecture comparison
# ---------------------------------------------------------------------------

def create_resnet50_model():
    """
    ResNet50 pretrained on ImageNet.

    Strips the global avg pool and FC from ResNet50, leaving layer4 output
    (N, 2048, H/32, W/32). MRNetBaseModel's adaptive_avg_pool2d then reduces
    spatial dims to (N, 2048, 1, 1) before max pooling across slices.

    Returns:
        MRNetBaseModel with feature_dim=2048
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove avgpool and fc — keep conv layers up to and including layer4
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    return MRNetBaseModel(backbone, feature_dim=2048)


# ---------------------------------------------------------------------------
# TorchXRayVision DenseNet121 — medical domain comparison
# ---------------------------------------------------------------------------

class _XRVBackbone(nn.Module):
    """
    Thin wrapper around TorchXRayVision's DenseNet121 feature extractor.

    TorchXRayVision models expect single-channel (grayscale) input.
    MRNet slices are loaded as grayscale and stacked into 3 channels,
    so we average across channels to recover the single-channel image.

    model.features is the DenseNet conv stack (no classifier), outputting
    (N, 1024, H', W') feature maps — compatible with MRNetBaseModel's
    adaptive_avg_pool2d -> max pool pipeline.
    """

    def __init__(self, xrv_features):
        super().__init__()
        self.features = xrv_features

    def forward(self, x):
        # x: (N, 3, H, W) — 3 identical grayscale channels
        x = x.mean(dim=1, keepdim=True)   # (N, 1, H, W)
        out = self.features(x)             # (N, 1024, H', W')
        # DenseNet feature extractor ends before BN+ReLU in the classifier block;
        # apply ReLU to ensure non-negative activations for max pooling.
        out = torch.relu(out)
        return out


def create_xrv_densenet_model():
    """
    DenseNet121 pretrained on combined chest X-ray datasets via TorchXRayVision.
    Weights: 'densenet121-res224-all' — trained on NIH ChestXray14 + CheXpert +
    MIMIC-CXR + PadChest (~600K chest X-ray images).

    This provides medical-domain pretraining with ZERO overlap with the
    MRNet (Stanford knee MRI) dataset.

    Requires: pip install torchxrayvision

    Returns:
        MRNetBaseModel with feature_dim=1024
    """
    try:
        import torchxrayvision as xrv
    except ImportError:
        raise ImportError(
            "torchxrayvision is required for the XRV DenseNet model.\n"
            "Install it with: pip install torchxrayvision"
        )

    xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    # xrv_model.features is the DenseNet conv feature extractor (no classifier head)
    backbone = _XRVBackbone(xrv_model.features)
    return MRNetBaseModel(backbone, feature_dim=1024)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

COMPARATIVE_ARCHITECTURES = {
    'resnet50':  create_resnet50_model,
    'xrv_dense': create_xrv_densenet_model,
}


def create_comparative_model(architecture='xrv_dense'):
    """
    Factory for comparative models.

    Args:
        architecture (str): One of:
            'resnet50'  — ResNet50 with ImageNet weights (feature_dim=2048)
            'xrv_dense' — DenseNet121 with chest X-ray weights (feature_dim=1024)

    Returns:
        MRNetBaseModel with the chosen backbone
    """
    if architecture not in COMPARATIVE_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(COMPARATIVE_ARCHITECTURES.keys())}"
        )
    print(f"Comparative architecture: {architecture}")
    return COMPARATIVE_ARCHITECTURES[architecture]()


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    dummy = torch.randn(1, 10, 3, 224, 224)   # 1 exam, 10 slices, 3ch, 224x224

    print("Testing ResNet50 (ImageNet)...")
    m1 = create_comparative_model('resnet50')
    out1 = m1(dummy)
    assert out1.shape == (1, 1), f"Expected (1,1), got {out1.shape}"
    print(f"  Output shape: {out1.shape}  OK")

    print("Testing XRV DenseNet121 (chest X-rays)...")
    m2 = create_comparative_model('xrv_dense')
    out2 = m2(dummy)
    assert out2.shape == (1, 1), f"Expected (1,1), got {out2.shape}"
    print(f"  Output shape: {out2.shape}  OK")

    print("\nAll comparative models OK.")
