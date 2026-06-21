from torchvision import models
import torch
import torch.nn as nn

resnet50 = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V1
)

backbone = nn.Sequential(
    *list(resnet50.children())[:-1]
)

x = torch.randn(2, 3, 224, 224)

out = backbone(x)

print("Output shape:", out.shape)