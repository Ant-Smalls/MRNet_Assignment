import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from modules.data_preprocessing_transformation import MRNetDataset, get_augmentation_transform
from modules.baseline_models import create_baseline_model


def parse_args():
    parser = argparse.ArgumentParser(description='Single-phase MRNet training')
    parser.add_argument('--condition', type=str, required=True, choices=['acl', 'meniscus', 'abnormal'])
    parser.add_argument('--plane', type=str, required=True, choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--architecture', type=str, default='baseline', choices=['baseline', 'comparative'])
    parser.add_argument('--data_mode', type=str, required=True, choices=['uncropped', 'cropped'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='src/data/')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training: {args.condition}-{args.plane}-{args.architecture}-{args.data_mode}")
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = MRNetDataset(
        root_dir=args.data_dir,
        condition=args.condition,
        plane=args.plane,
        split='train',
        data_mode=args.data_mode,
        transform=get_augmentation_transform(is_training=True)
    )
    val_dataset = MRNetDataset(
        root_dir=args.data_dir,
        condition=args.condition,
        plane=args.plane,
        split='val',
        data_mode=args.data_mode,
        transform=None
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model and optimizer
    model = create_baseline_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_dataset.pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{args.checkpoint_dir}/{args.condition}_{args.plane}_{args.architecture}_{args.data_mode}_best.pth"
    
    best_auc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for exam, label in train_loader:
            exam, label = exam.to(device), label.to(device)
            
            optimizer.zero_grad()
            logits = model(exam)
            loss = criterion(logits, label.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for exam, label in val_loader:
                exam = exam.to(device)
                logits = model(exam)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(label.numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val AUC: {val_auc:.4f}")
        
        # Save best checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Saved best checkpoint (AUC: {val_auc:.4f})")
    
    print(f"\nTraining complete - Best Val AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
