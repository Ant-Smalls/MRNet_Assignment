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
        
        # Max pooling across the slice dimension (picks the "loudest" slice)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Binary classification head: 512 thoughts -> 1 final score
        self.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input of shape (batch_size, num_slices, 3, H, W)
                             For batch_size=1: (1, S, 3, H, W)
        
        Returns:
            torch.Tensor: Logit of shape (batch_size, 1)
        
        Steps:
        1. Reshape so all slices are processed individually by ResNet18
        2. Run every slice through the ResNet18 backbone
        3. Reshape back to group slices by exam
        4. Max pool to find the "loudest" signal across all slices
        5. Pass through our new mouth (fc) to get the final score
        """
        B, S, C, H, W = x.shape  # B=batch(1), S=slices, C=3channels, H=height, W=width
        
        # Step 1: Flatten batch+slice dims so ResNet sees each slice as a separate image
        x = x.view(B * S, C, H, W)          # (B*S, 3, H, W)
        
        # Step 2: Pass all slices through the ResNet18 backbone
        x = self.backbone(x)                 # (B*S, 512, 1, 1)  <- ResNet adds spatial dims
        x = x.squeeze(-1).squeeze(-1)        # (B*S, 512)  <- remove the extra 1x1 spatial dims
        
        # Step 3: Reshape back to group slices per exam
        x = x.view(B, S, 512)               # (B, S, 512)
        
        # Step 4: Max pool across the slice dimension - "loudest voice wins"
        # permute to (B, 512, S) because AdaptiveMaxPool1d pools over last dim
        x = x.permute(0, 2, 1)              # (B, 512, S)
        x = self.global_pool(x)             # (B, 512, 1)
        x = x.squeeze(-1)                   # (B, 512)
        
        # Step 5: Binary classification - output a single score per exam
        x = self.fc(x)                      # (B, 1)
        return x
    
    def freeze_backbone(self):
        """
        Freeze all backbone parameters (for Phase 1 training).
        
        This prevents ResNet18's pre-learned knowledge from being corrupted
        while our new classification head is still learning from scratch.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, num_layers=None):
        """
        Unfreeze backbone parameters (for Phase 2 fine-tuning).
        
        Once the classification head is trained, we unfreeze the backbone
        and allow the entire model to be fine-tuned.
        
        Args:
            num_layers (int, optional): If specified, only unfreeze last N child blocks.
                                       If None, unfreeze all.
        """
        if num_layers is None:
            # Unfreeze the entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Only unfreeze the last num_layers children of the backbone
            children = list(self.backbone.children())
            for child in children[-num_layers:]:
                for param in child.parameters():
                    param.requires_grad = True
    
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
    """
    # Load ResNet18 with ImageNet weights (it already knows how to see edges/shapes)
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Surgical step: remove the final classification layer (the "1000-word mouth")
    # children()[:-1] means "take every layer EXCEPT the last one"
    # AdaptiveAvgPool2d is the last spatial pooling; we keep it for feature compression
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    
    # Build our MRNet model with this pre-trained backbone
    model = MRNetBaseModel(backbone)
    return model


if __name__ == '__main__':
    print("Testing Baseline Model (Role 2)...")
    
    # Step A: Build the model
    model = create_baseline_model()
    print(f"✅ Model created successfully.")
    
    # Step B: Freeze the backbone (Phase 1 training mode)
    model.freeze_backbone()
    frozen_count = len(model.get_trainable_params())
    print(f"✅ Backbone frozen. Trainable parameters: {frozen_count} (only the new mouth/fc layer)")
    
    # Step C: Unfreeze the backbone (Phase 2 fine-tuning mode)
    model.unfreeze_backbone()
    unfrozen_count = len(model.get_trainable_params())
    print(f"✅ Backbone unfrozen. Trainable parameters: {unfrozen_count} (all layers)")
    
    # Step D: Do a dummy forward pass to prove the shapes are all correct
    # Simulate 1 exam with 22 slices, 3 channels (RGB-like), 256x256 pixels
    dummy_input = torch.zeros(1, 22, 3, 256, 256)  # (B=1, S=22, C=3, H=256, W=256)
    model.eval()  # Set to evaluation mode (disables dropout etc.)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✅ Forward pass success! Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
    print(f"   (Output should be shape [1, 1] — one score per exam)")
