import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate


class MRNetDataset(Dataset):
    """MRNet Dataset for knee MRI classification. Loads patient splits from JSON,
    applies min-max normalization, and handles variable-length 3D volumes."""
    
    def __init__(self, root_dir, condition, plane, split, data_mode, transform=None):
        self.root_dir = root_dir
        self.condition = condition.lower()
        self.plane = plane.lower()
        self.split = split.lower()
        self.data_mode = data_mode.lower()
        self.transform = transform
        
        self.file_paths = []
        self.labels = []
        
        # Construct data directory path based on cropped/uncropped mode
        # Both train/val use mrnet/train/ since we split the MRNet training data
        if self.data_mode == 'uncropped':
            data_dir = f"{root_dir}/mrnet/train/{plane}/"
        else:
            data_dir = f"{root_dir}/mrnet_cropped/train/{plane}/"
        
        label_csv = f"{root_dir}/mrnet/train-{condition}.csv"
        
        if split in ['train', 'val']:
            # Load patient IDs from our custom split
            split_json = f"{root_dir}/patient_splits_training.json"
            with open(split_json) as f:
                patient_ids = json.load(f)[split]
            
            # Filter CSV labels to only include patients in this split
            df = pd.read_csv(label_csv, header=0)
            df = df[df.iloc[:, 0].isin(patient_ids)]
            
            # Build file paths (zero-padded to 4 digits: 65 -> 0065.npy)
            for patient_id, label in df.itertuples(index=False):
                filename = f"{int(patient_id):04d}.npy"
                path = os.path.join(data_dir, filename)
                if os.path.exists(path):
                    self.file_paths.append(path)
                    self.labels.append(int(label))
            
            # Calculate pos_weight for BCEWithLogitsLoss (training only)
            if split == 'train':
                num_positive = sum(self.labels)
                num_negative = len(self.labels) - num_positive
                self.pos_weight = torch.tensor(num_negative / num_positive if num_positive > 0 else 1.0)
                print(f"[{condition}-{plane}-{split}] pos_weight: {self.pos_weight:.3f} "
                      f"({num_positive} pos, {num_negative} neg)")
            else:
                self.pos_weight = torch.tensor(1.0)
        else:
            self.pos_weight = torch.tensor(1.0)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Returns: (exam_tensor, label) where exam_tensor has shape (num_slices, 3, H, W)"""
        volume = np.load(self.file_paths[idx])
        
        # Handle 4D volumes by extracting first channel
        if volume.ndim == 4 and volume.shape[1] == 3:
            volume = volume[:, 0, :, :]
        
        # Min-max normalization to [0, 1]
        volume = volume.astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Apply augmentations per-slice (if training)
        if self.transform:
            volume = np.stack([self.transform(slice_2d) for slice_2d in volume])
        
        # Convert to tensor and stack to 3 channels for ResNet
        exam_tensor = torch.from_numpy(volume).float()
        exam_tensor = exam_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return exam_tensor, label


def get_augmentation_transform(is_training=True):
    """Medical imaging augmentations (brightness/contrast jitter, rotations ±10°).
    Returns: callable that takes and returns a 2D numpy array (single slice)."""
    if is_training:
        def augment(slice_2d):
            # Brightness/contrast jitter (50% probability)
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-0.1, 0.1)
                slice_2d = np.clip(alpha * slice_2d + beta, 0, 1)
            
            # Random rotation ±10° (50% probability)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                slice_2d = rotate(slice_2d, angle, reshape=False, mode='nearest')
            
            return slice_2d
        return augment
    else:
        return lambda x: x


class ValidMRNetDataset(Dataset):
    """Minimal dataset class specifically for the true validation set."""
    def __init__(self, root_dir, condition, plane, data_mode):
        self.condition = condition.lower()
        self.plane = plane.lower()
        
        # Point to the valid directory instead of train
        if data_mode == 'uncropped':
            self.data_dir = os.path.join(root_dir, f"mrnet/valid/{self.plane}/")
        else:
            self.data_dir = os.path.join(root_dir, f"mrnet_cropped/valid/{self.plane}/")
            
        label_csv = os.path.join(root_dir, f"mrnet/valid-{self.condition}.csv")
        
        # Read CSV and clean up the underscores (e.g., '_1130' -> 1130)
        df = pd.read_csv(label_csv, header=None, names=['id', 'label'])
        df['id'] = df['id'].astype(str).str.replace('_', '').astype(int)
        df['label'] = df['label'].astype(str).str.replace('_', '').astype(int)
        
        self.file_paths = []
        self.labels = []
        
        for patient_id, label in df.itertuples(index=False):
            filename = f"{int(patient_id):04d}.npy"
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                self.file_paths.append(path)
                self.labels.append(int(label))
                
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        volume = np.load(self.file_paths[idx])
        
        # Handle 4D volumes by extracting first channel
        if volume.ndim == 4 and volume.shape[1] == 3:
            volume = volume[:, 0, :, :]
            
        # Min-max normalization to [0, 1]
        volume = volume.astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Convert to tensor and stack to 3 channels for ResNet
        exam_tensor = torch.from_numpy(volume).float()
        exam_tensor = exam_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return exam_tensor, label

