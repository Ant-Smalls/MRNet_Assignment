"""
Role 4: Training Pipeline & Hyperparameter Tuning

Two-phase training for SLURM batch jobs.
Usage: python train.py --condition acl --plane sagittal --architecture baseline
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.data_preprocessing_transformation import MRNetDataset, get_augmentation_transform
from modules.baseline_models import create_baseline_model
from modules.comparative_models import create_comparative_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True, choices=['acl', 'meniscus', 'abnormal'])
    parser.add_argument('--plane', type=str, required=True, choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--architecture', type=str, default='baseline')
    parser.add_argument('--epochs_phase1', type=int, default=10)
    parser.add_argument('--epochs_phase2', type=int, default=20)
    parser.add_argument('--lr_phase1', type=float, default=1e-3)
    parser.add_argument('--lr_phase2', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='src/data/mrnet/')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    return parser.parse_args()


def create_data_loaders(args):
    """
    Create train and validation dataloaders.
    Returns: (train_loader, val_loader, pos_weight)
    
    TODO: Create datasets (80/20 split), get pos_weight, create DataLoaders
    """
    pass


def create_model(args):
    """
    Create model based on architecture.
    TODO: Call create_baseline_model() or create_comparative_model(args.architecture)
    """
    pass


def train_phase(model, train_loader, val_loader, optimizer, criterion, 
                num_epochs, phase_name, checkpoint_dir, writer):
    """
    Train for one phase, return best validation loss.
    
    TODO: Loop epochs, train, validate, log to TensorBoard, save best checkpoint
    """
    pass


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train one epoch, return average loss.
    TODO: Set train mode, loop batches, forward/backward pass
    """
    pass


def validate(model, dataloader, criterion, device):
    """
    Validate model, return (avg_loss, auc_score).
    TODO: Set eval mode, compute loss and AUC
    """
    pass


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save checkpoint.
    TODO: Save model/optimizer state_dict, epoch, loss
    """
    pass


def main():
    """Main training function."""
    args = parse_args()
    model_name = f"{args.condition}_{args.plane}_{args.architecture}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {model_name} on {device}")
    
    # TODO: Create dataloaders and model
    train_loader, val_loader, pos_weight = create_data_loaders(args)
    model = create_model(args).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    writer = SummaryWriter(log_dir=os.path.join('runs', model_name))
    
    # Phase 1: Frozen backbone
    print("\nPhase 1: Frozen backbone")
    model.freeze_backbone()
    optimizer1 = optim.Adam(model.get_trainable_params(), lr=args.lr_phase1)
    best_loss1 = train_phase(model, train_loader, val_loader, optimizer1, criterion,
                             args.epochs_phase1, 'phase1', checkpoint_dir, writer)
    
    # Phase 2: Fine-tuning
    print("\nPhase 2: Fine-tuning")
    model.unfreeze_backbone()
    optimizer2 = optim.Adam(model.get_trainable_params(), lr=args.lr_phase2)
    best_loss2 = train_phase(model, train_loader, val_loader, optimizer2, criterion,
                             args.epochs_phase2, 'phase2', checkpoint_dir, writer)
    
    writer.close()
    print(f"\nComplete! Best loss - Phase1: {best_loss1:.4f}, Phase2: {best_loss2:.4f}")


if __name__ == '__main__':
    main()
