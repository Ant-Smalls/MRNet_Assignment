"""
Role 4: Training Pipeline & Hyperparameter Tuning

Configurable single-phase or two-phase training for SLURM batch jobs.
  Single-phase: All layers train together from start
  Two-phase: Phase 1 (frozen backbone) -> Phase 2 (full fine-tuning)

Usage: python train.py --condition acl --plane sagittal --architecture baseline --data_mode uncropped
"""

#imports
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

from modules.data_preprocessing_transformation import MRNetDataset, get_augmentation_transform
from modules.baseline_models import create_baseline_model
from modules.comparative_models import create_comparative_model


#control training script without editing code
#example usage: python train.py --condition acl --plane sagittal --architecture baseline --data_mode uncropped
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition',       type=str,   required=True, choices=['acl', 'meniscus', 'abnormal'])
    parser.add_argument('--plane',           type=str,   required=True, choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--architecture',    type=str,   default='baseline', choices=['baseline', 'comparative'])
    parser.add_argument('--data_mode',       type=str,   required=True, choices=['uncropped', 'cropped'])
    parser.add_argument('--training_mode',   type=str,   default='two-phase', choices=['single-phase', 'two-phase'],
                        help='Training strategy: single-phase (all layers) or two-phase (frozen then full)')
    parser.add_argument('--epochs',          type=int,   default=30,
                        help='Number of epochs for single-phase training')
    parser.add_argument('--epochs_phase1',   type=int,   default=10,
                        help='Number of epochs for phase 1 (frozen backbone) in two-phase training')
    parser.add_argument('--epochs_phase2',   type=int,   default=20,
                        help='Number of epochs for phase 2 (full fine-tuning) in two-phase training')
    parser.add_argument('--lr',              type=float, default=1e-4,
                        help='Learning rate for single-phase training')
    parser.add_argument('--lr_phase1',       type=float, default=1e-3,
                        help='Learning rate for phase 1 in two-phase training')
    parser.add_argument('--lr_phase2',       type=float, default=1e-4,
                        help='Learning rate for phase 2 in two-phase training')
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--patience',        type=int,   default=5,
                        help='Early stopping patience (epochs without AUC improvement)')
    parser.add_argument('--optimizer',       type=str,   default='adamw', choices=['adamw', 'sgd'],
                        help='Optimizer to use for training')
    parser.add_argument('--batch_size',      type=int,   default=1)
    parser.add_argument('--data_dir',        type=str,   default='src/data/',
                        help='Root directory containing mrnet/ and mrnet_cropped/ folders')
    parser.add_argument('--checkpoint_dir',  type=str,   default='checkpoints/')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_data_loaders(args):
    """
    Create train and validation DataLoaders using patient_splits.json.
    Uses pre-defined patient-level splits for consistent validation across experiments.

    Returns: (train_loader, val_loader, pos_weight)
      - pos_weight is a scalar tensor for BCEWithLogitsLoss
    """
    # Training dataset with augmentation
    train_dataset = MRNetDataset(
        root_dir=args.data_dir,
        condition=args.condition,
        plane=args.plane,
        split='train',
        data_mode=args.data_mode,
        transform=get_augmentation_transform(is_training=True),
    )

    # Validation dataset without augmentation
    val_dataset = MRNetDataset(
        root_dir=args.data_dir,
        condition=args.condition,
        plane=args.plane,
        split='val',
        data_mode=args.data_mode,
        transform=None,
    )

    # batch_size=1 because MRI volumes have variable slice counts
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    # pos_weight from the training dataset label distribution
    # BCEWithLogitsLoss uses this to penalise missing the minority class
    pos_weight = train_dataset.pos_weight

    print(f"Dataset sizes — train: {len(train_dataset)}, val: {len(val_dataset)}")
    print(f"pos_weight: {pos_weight.item():.3f}  "
          f"(higher = more imbalanced toward negatives)")

    return train_loader, val_loader, pos_weight


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model(args):
    """Instantiate the correct architecture."""
    if args.architecture == 'baseline':
        print("Architecture: ResNet18 (ImageNet weights)")
        return create_baseline_model()
    else:
        print("Architecture: ResNet50 (RadImageNet weights)")
        return create_comparative_model()


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def create_optimizer(args, params, lr):
    """
    Create optimizer based on --optimizer flag.
    AdamW: adaptive learning rate, good default for fine-tuning pretrained CNNs.
    SGD with momentum: simpler update rule, can generalise better with enough tuning.
    """
    if args.optimizer == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    else:
        return optim.SGD(params, lr=lr, weight_decay=args.weight_decay, 
                         momentum=0.9, nesterov=True)


# ---------------------------------------------------------------------------
# Core training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    One full pass over the training set.
    Returns average loss for the epoch.
    """
    #set model to train mode
    model.train()
    total_loss = 0.0

    for exam, label in tqdm(dataloader, desc="Training", leave=False):
        # exam:  (1, num_slices, 3, H, W)  — batch size is always 1
        # label: (1,)
        #move data to GPU
        exam  = exam.to(device)
        label = label.float().to(device)

        #clear gradiants from previous scan
        optimizer.zero_grad()
        logits = model(exam).squeeze(1)   # (1,) → scalar-ish
        loss   = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Run the validation set.
    Returns (avg_loss, auc_score).
    AUC is the primary metric used for early stopping and checkpointing.
    """
    #evaluation mode
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    #dont track gradiants
    with torch.no_grad():
        for exam, label in tqdm(dataloader, desc="Validation", leave=False):
            exam  = exam.to(device)
            label = label.float().to(device)

            logits = model(exam).squeeze(1)
            loss   = criterion(logits, label)
            total_loss += loss.item()

            #convert logits -> probabilities for AUC
            prob = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(prob.tolist())
            all_labels.extend(label.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)

    # AUC requires at least one positive and one negative sample
    unique_labels = set(all_labels)
    if len(unique_labels) < 2:
        auc = 0.5   # undefined —> return chance level
    else:
        auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, auc


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, loss, auc, checkpoint_path):
    """Save model state, optimizer state, and training metadata."""
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss':             loss,
        'val_auc':              auc,
    }, checkpoint_path)


# ---------------------------------------------------------------------------
# Phase trainer
# ---------------------------------------------------------------------------

def train_phase(model, train_loader, val_loader, optimizer, criterion,
                num_epochs, phase_name, checkpoint_dir, writer, patience, device):
    """
    Train for one phase with early stopping on validation AUC.

    Saves the best checkpoint (by AUC) to:
        {checkpoint_dir}/best_{phase_name}.pth

    Returns best validation AUC achieved during this phase.
    """
    best_auc = 0.0
    epochs_no_improve = 0
    checkpoint_path = os.path.join(checkpoint_dir, f'best_{phase_name}.pth')

    #scheduer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'max',
        factor = 0.5,
        patience = 3
    )
    #for each epoch, train and validate 
    for epoch in range(1, num_epochs + 1):
        train_loss          = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc   = validate(model, val_loader, criterion, device)

        #step scheduler
        scheduler.step(val_auc)
        #log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar(f'{phase_name}/learning_rate', current_lr, epoch)

        # TensorBoard logging
        writer.add_scalar(f'{phase_name}/train_loss', train_loss, epoch)
        writer.add_scalar(f'{phase_name}/val_loss',   val_loss,   epoch)
        writer.add_scalar(f'{phase_name}/val_auc',    val_auc,    epoch)

        print(f"  [{phase_name}] Epoch {epoch}/{num_epochs} — "
              f"train_loss: {train_loss:.4f}  "
              f"val_loss: {val_loss:.4f}  "
              f"val_auc: {val_auc:.4f}")

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss, val_auc, checkpoint_path)
            print(f" New best AUC: {best_auc:.4f} — checkpoint saved")
        else:
            epochs_no_improve += 1
            print(f" No improvement ({epochs_no_improve}/{patience})")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"  Early stopping triggered after {epoch} epochs.")
            break

    # Reload best weights before returning so Phase 2 starts from the best Phase 1 state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded best {phase_name} weights (AUC: {best_auc:.4f})")

    return best_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args       = parse_args()
    #creates a unique model and folder name based on the training parameters
    model_name = f"{args.condition}_{args.plane}_{args.architecture}_{args.data_mode}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training: {model_name}")
    print(f"Training mode: {args.training_mode}")
    print(f"Device:   {device}\n")

    # Save config for reproducibility — Role 5 can read this to know what was used
    config = vars(args)
    config['model_name'] = model_name
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Data, model, loss, logging
    train_loader, val_loader, pos_weight = create_data_loaders(args)
    model     = create_model(args).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    writer    = SummaryWriter(log_dir=os.path.join('runs', model_name))

    if args.training_mode == 'two-phase':
        # Two-phase training: Phase 1 (frozen backbone) -> Phase 2 (full fine-tuning)
        
        # Phase 1: frozen backbone, train head only
        print("Phase 1: Frozen backbone (head only)")
        model.freeze_backbone()
        optimizer1 = create_optimizer(args, model.get_trainable_params(), args.lr_phase1)
        best_auc1 = train_phase(
            model, train_loader, val_loader,
            optimizer1, criterion,
            args.epochs_phase1, 'phase1',
            checkpoint_dir, writer,
            args.patience, device,
        )

        # Phase 2: full fine tuning
        print("\nPhase 2: Full fine-tuning (all layers)")
        model.unfreeze_backbone()
        optimizer2 = create_optimizer(args, model.parameters(), args.lr_phase2)
        best_auc2 = train_phase(
            model, train_loader, val_loader,
            optimizer2, criterion,
            args.epochs_phase2, 'phase2',
            checkpoint_dir, writer,
            args.patience, device,
        )

        # Save final summary so Role 5 can quickly read results without loading checkpoints
        summary = {
            'model_name':        model_name,
            'training_mode':     'two-phase',
            'best_auc_phase1':   best_auc1,
            'best_auc_phase2':   best_auc2,
        }
        
        print(f"\nBest AUC — Phase 1: {best_auc1:.4f}  Phase 2: {best_auc2:.4f}")
        
    else:
        # Single-phase training: All layers train from start
        print("Single-phase training: All layers unfrozen from start")
        
        # Use --lr if provided, otherwise default to lr_phase2
        lr = args.lr
        optimizer = create_optimizer(args, model.parameters(), lr)
        best_auc = train_phase(
            model, train_loader, val_loader,
            optimizer, criterion,
            args.epochs, 'single_phase',
            checkpoint_dir, writer,
            args.patience, device,
        )
        
        # Save final summary
        summary = {
            'model_name':      model_name,
            'training_mode':   'single-phase',
            'best_auc':        best_auc,
        }
        
        print(f"\nBest AUC: {best_auc:.4f}")
    
    # Save summary.json
    with open(os.path.join(checkpoint_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    writer.close()
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()