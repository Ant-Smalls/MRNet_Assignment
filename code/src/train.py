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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

import sys
sys.path.insert(0, os.path.dirname(__file__))

from modules.data_preprocessing_transformation import MRNetDataset, get_augmentation_transform, stack_channels
from modules.baseline_models import create_baseline_model
try:
    from modules.comparative_models import create_comparative_model
except ImportError:
    create_comparative_model = None  # Role 3 not implemented yet


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
    parser.add_argument('--checkpoint_dir', type=str, default='local_learning_hub/checkpoints/')
    return parser.parse_args()


def create_data_loaders(args):
    """
    Create train and validation dataloaders.
    Returns: (train_loader, val_loader, pos_weight)
    """
    train_dataset = MRNetDataset(
        root_dir=args.data_dir,
        condition=args.condition,
        plane=args.plane,
        split='train',
        transform=get_augmentation_transform(is_training=True)
    )
    val_dataset = MRNetDataset(
        root_dir=args.data_dir,
        condition=args.condition,
        plane=args.plane,
        split='valid',
        transform=None  # No augmentation on validation!
    )
    # batch_size=1 is mandatory — MRIs have different numbers of slices
    # so we cannot stack them into a normal batch
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)
    return train_loader, val_loader, train_dataset.pos_weight


def create_model(args):
    """Create model based on architecture argument."""
    if args.architecture == 'baseline':
        return create_baseline_model()
    elif create_comparative_model is not None:
        return create_comparative_model(args.architecture)
    else:
        raise ValueError(f"Comparative models not yet implemented. Use --architecture baseline")


def train_phase(model, train_loader, val_loader, optimizer, criterion,
                num_epochs, phase_name, checkpoint_dir, writer):
    """
    Train for one phase (e.g., Phase 1=frozen backbone, Phase 2=fine-tuning).
    Returns the best validation loss seen during this phase.
    """
    best_val_loss = float('inf')  # Start with infinity — anything is better than this
    device = next(model.parameters()).device  # Find out which device the model is on

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # ---- Validate ----
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(f"  [{phase_name}] Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Log to TensorBoard if available
        if writer and TENSORBOARD_AVAILABLE:
            writer.add_scalar(f'{phase_name}/train_loss', train_loss, epoch)
            writer.add_scalar(f'{phase_name}/val_loss', val_loss, epoch)
            writer.add_scalar(f'{phase_name}/val_auc', val_auc, epoch)

        # Save the model if this is the best it has ever done
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'best_{phase_name}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

    return best_val_loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train one epoch, return average loss.
    The model looks at each MRI, makes a guess, measures how wrong it is,
    and adjusts its brain connections to do better next time.
    """
    model.train()  # Tell the model "you are in learning mode" (enables dropout etc.)
    total_loss = 0.0

    # tqdm wraps the dataloader to show a live progress bar in the console
    progress = tqdm(dataloader, desc="  Training", unit="exam", leave=False)

    for exam, label in progress:
        # Stack grayscale (S,1,H,W) -> (S,3,H,W) so ResNet18 accepts it
        exam = stack_channels(exam.squeeze(0))   # remove the batch dim added by DataLoader
        exam = exam.unsqueeze(0).to(device)       # add batch dim back: (1, S, 3, H, W)
        label = label.to(device)                  # (1, 1)

        optimizer.zero_grad()           # Clear any leftover gradients from last round
        output = model(exam)            # Forward pass: get the model's guess
        loss = criterion(output, label) # Calculate punishment (BCEWithLogitsLoss)
        loss.backward()                 # Backprop: figure out who was responsible
        optimizer.step()                # Nudge the brain weights in the right direction

        total_loss += loss.item()
        # Update the progress bar to show the current running loss
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)  # Average loss across all exams


def validate(model, dataloader, criterion, device):
    """
    Run on validation exams (never seen during training).
    Returns: (avg_loss, auc_score)
    """
    model.eval()   # Tell the model "you are in test mode" (no learning allowed)
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():  # "no_grad" means: don't track anything, just observe
        progress = tqdm(dataloader, desc="  Validating", unit="exam", leave=False)
        for exam, label in progress:
            exam = stack_channels(exam.squeeze(0))
            exam = exam.unsqueeze(0).to(device)
            label = label.to(device)

            output = model(exam)
            loss = criterion(output, label)
            total_loss += loss.item()

            # Sigmoid converts the raw score into a proper 0-1 probability
            prob = torch.sigmoid(output).item()
            all_probs.append(prob)
            all_labels.append(label.item())

    avg_loss = total_loss / len(dataloader)
    # AUC measures how well the model separates tears from normal knees
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5  # if all labels are the same, AUC is undefined — default to random
    return avg_loss, auc


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save a snapshot of the model's brain at this moment.
    If we save when the model is at its best, we can restore it later.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"   💾 Saved best checkpoint (loss: {loss:.4f}) -> {checkpoint_path}")


def main():
    """Main training function."""
    args = parse_args()
    model_name = f"{args.condition}_{args.plane}_{args.architecture}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Detect the best available device: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')   # Your M2 Pro's built-in GPU!
    else:
        device = torch.device('cpu')
    print(f"Training {model_name} on {device}")

    train_loader, val_loader, pos_weight = create_data_loaders(args)
    model = create_model(args).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    writer = SummaryWriter(log_dir=os.path.join('runs', model_name)) if TENSORBOARD_AVAILABLE else None

    # Phase 1: Train only the new classification head (backbone frozen)
    print("\nPhase 1: Frozen backbone — only the classification head learns")
    model.freeze_backbone()
    optimizer1 = optim.Adam(model.get_trainable_params(), lr=args.lr_phase1)
    best_loss1 = train_phase(model, train_loader, val_loader, optimizer1, criterion,
                             args.epochs_phase1, 'phase1', checkpoint_dir, writer)

    # Phase 2: Fine-tune the whole brain together
    print("\nPhase 2: Fine-tuning — the whole brain learns together")
    model.unfreeze_backbone()
    optimizer2 = optim.Adam(model.get_trainable_params(), lr=args.lr_phase2)
    best_loss2 = train_phase(model, train_loader, val_loader, optimizer2, criterion,
                             args.epochs_phase2, 'phase2', checkpoint_dir, writer)

    if writer:
        writer.close()
    print(f"\n✅ Complete! Best loss — Phase1: {best_loss1:.4f}, Phase2: {best_loss2:.4f}")


if __name__ == '__main__':
    main()
