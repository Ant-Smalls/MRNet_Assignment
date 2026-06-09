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
        
        # TODO: Load file paths from {root_dir}/{split}/{plane}/
        self.file_paths = []  # List of .npy file paths
        
        # TODO: Load labels from CSV file ({split}-{condition}.csv)
        self.labels = []  # List of binary labels (0 or 1)
        
        # TODO: Compute pos_weight = num_negative / num_positive
        self.pos_weight = torch.tensor(1.0)
        
    def __len__(self):
        # TODO: Return number of exams
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            exam_tensor (torch.Tensor): Shape (num_slices, H, W)
            label (torch.Tensor): Scalar (0 or 1)
        """
        # TODO: Load .npy file
        # TODO: Normalize to [0, 1]
        # TODO: Apply transforms if training
        # TODO: Convert to tensor and return with label
        pass


def get_augmentation_transform(is_training=True):
    """
    Returns augmentation pipeline.
    
    Training: Add ColorJitter for contrast/brightness (NO horizontal flips!)
    Val/Test: Only normalization
    """
    if is_training:
        # TODO: Add transforms.ColorJitter(brightness=0.2, contrast=0.2)
        pass
    else:
        # TODO: Only normalization transforms
        pass


def stack_channels(exam_tensor):
    """
    Convert 1-channel to 3-channel: (S, H, W) -> (S, 3, H, W)
    
    TODO: Repeat grayscale channel 3 times for ImageNet models
    """
    pass


if __name__ == '__main__':
    # TODO: Test dataset loading
    # dataset = MRNetDataset('src/data/mrnet/', 'acl', 'sagittal', 'train')
    # print(f"Dataset size: {len(dataset)}, pos_weight: {dataset.pos_weight}")
    pass
