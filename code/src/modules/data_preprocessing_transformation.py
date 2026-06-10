"""
Role 1: Data Preprocessing & Transformations

Dataset class and utilities for loading MRNet knee MRI data.
Handles variable-length 3D tensors, medical imaging augmentations, and class imbalance.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


class MRNetDataset(Dataset):
    """
    MRNet Dataset for knee MRI classification.
    
    Args:
        root_dir (str): Path to data directory (e.g., 'src/data/mrnet/')
        condition (str): 'acl', 'meniscus', or 'abnormal'
        plane (str): 'axial', 'coronal', or 'sagittal'
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Augmentation transforms
        
    Attributes:
        pos_weight (torch.Tensor): Class imbalance weight (num_negative / num_positive)
    """
    
    def __init__(self, root_dir, condition, plane, split='train', transform=None):
        self.root_dir = root_dir
        self.condition = condition.lower()
        self.plane = plane.lower()
        self.split = split.lower()
        self.transform = transform
        
        # Construct paths
        self.plane_dir = os.path.join(self.root_dir, self.split, self.plane)
        csv_path = os.path.join(self.root_dir, f"{self.split}-{self.condition}.csv")
        
        # Load labels from CSV
        df = pd.read_csv(csv_path, header=None, names=["patient_id", "label"])
        # Skip the dummy header row if present (_0000)
        df = df[~df["patient_id"].astype(str).str.startswith("_")]
        
        # Ensure patient_id matches the zero-padded filenames (e.g., "0005.npy")
        df["patient_id"] = df["patient_id"].astype(str).str.zfill(4)
        df["label"] = df["label"].astype(float)
        
        # Keep only exams that actually exist in the folder
        existing_files = set(f.replace('.npy', '') for f in os.listdir(self.plane_dir) if f.endswith('.npy'))
        df = df[df["patient_id"].isin(existing_files)]
        
        self.file_paths = [os.path.join(self.plane_dir, f"{pid}.npy") for pid in df["patient_id"]]
        self.labels = df["label"].tolist()
        
        # Compute pos_weight = num_negative / num_positive (only for training)
        if self.split == 'train':
            num_positive = sum(self.labels)
            num_negative = len(self.labels) - num_positive
            # If there are no positives, avoid division by zero
            weight = num_negative / num_positive if num_positive > 0 else 1.0
            self.pos_weight = torch.tensor([weight], dtype=torch.float32)
        else:
            self.pos_weight = torch.tensor([1.0], dtype=torch.float32)
        
    def __len__(self):
        # TODO: Return number of exams
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            exam_tensor (torch.Tensor): Shape (num_slices, H, W)
            label (torch.Tensor): Scalar (0 or 1)
        """
        # 1. Load the MRI volume (it's a 3D block of numbers from 0 to 255)
        volume = np.load(self.file_paths[idx])
        
        # 2. Normalize to [0, 1] so the neural network can understand it
        volume = volume.astype(np.float32) / 255.0
        
        # 3. Convert to a PyTorch Tensor
        exam_tensor = torch.FloatTensor(volume)
        label_tensor = torch.FloatTensor([self.labels[idx]])
        
        # 3.5. PyTorch expects images to have a "color channel" dimension.
        # Since these are grayscale, we add a dimension of 1: (num_slices, 1, H, W)
        exam_tensor = exam_tensor.unsqueeze(1)
        
        # 4. Apply clinical augmentations (brightness/contrast) if training
        if self.transform:
            exam_tensor = self.transform(exam_tensor)
            
        return exam_tensor, label_tensor


def get_augmentation_transform(is_training=True):
    """
    Returns augmentation pipeline.
    
    Training: Add ColorJitter for contrast/brightness (NO horizontal flips!)
    Val/Test: Only normalization
    """
    if is_training:
        # We only tweak brightness and contrast. NO horizontal flipping!
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
    else:
        # Validation/Test should NOT be augmented
        return None


def stack_channels(exam_tensor):
    """
    Convert 1-channel to 3-channel: (S, H, W) -> (S, 3, H, W)
    
    Input shape: (S, 1, H, W)
    The pre-trained model expects color images (S, 3, H, W).
    We repeat the grayscale channel 3 times to fake "RGB" color.
    """
    return exam_tensor.repeat(1, 3, 1, 1)


if __name__ == '__main__':
    print("Testing Data Preprocessing (Role 1)...")
    dataset = MRNetDataset('code/src/data/mrnet/', 'acl', 'sagittal', 'train', transform=get_augmentation_transform(is_training=True))
    
    print(f"✅ Loaded Dataset. Size: {len(dataset)} exams.")
    print(f"✅ Calculated pos_weight for ACL: {dataset.pos_weight.item():.2f}")
    
    # Grab the first exam
    exam, label = dataset[0]
    print(f"✅ Loaded Exam 0. Shape: {exam.shape}")
    print(f"✅ Normalized min/max: {exam.min().item():.2f} / {exam.max().item():.2f}")
    
    # Stack channels
    color_exam = stack_channels(exam)
    print(f"✅ Stacked channels for ResNet. New Shape: {color_exam.shape}")
