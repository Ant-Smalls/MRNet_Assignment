import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)
from tqdm import tqdm

from modules.baseline_models import create_baseline_model
from modules.data_preprocessing_transformation import ValidMRNetDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Minimal MRNet Evaluation')
    parser.add_argument('--condition', type=str, required=True, choices=['acl', 'meniscus', 'abnormal'])
    parser.add_argument('--plane', type=str, required=True, choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--data_mode', type=str, required=True, choices=['uncropped', 'cropped'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='src/data/')
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Evaluating: {args.condition}-{args.plane}-{args.data_mode}")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Create dataset and dataloader
    val_dataset = ValidMRNetDataset(
        root_dir=args.data_dir,
        condition=args.condition,
        plane=args.plane,
        data_mode=args.data_mode
    )
    
    if len(val_dataset) == 0:
        print("Error: No validation data found! Check your data_dir and data_mode.")
        return
        
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model and load weights
    model = create_baseline_model().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle if the checkpoint is a dict with 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    val_preds = []
    val_labels = []
    
    # Evaluation loop
    with torch.no_grad():
        for exam, label in tqdm(val_loader, desc="Evaluating"):
            exam = exam.to(device)
            logits = model(exam)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            val_preds.extend(probs)
            val_labels.extend(label.numpy())
    # AUC        
    val_auc = roc_auc_score(val_labels, val_preds)
    
    # Binary metrics (using 0.5 threshold)
    val_preds_binary = (np.array(val_preds) >= 0.5).astype(int)
    
    val_accuracy = accuracy_score(val_labels, val_preds_binary)
    val_precision = precision_score(val_labels, val_preds_binary, zero_division=0)
    val_recall = recall_score(val_labels, val_preds_binary, zero_division=0) 
    val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)
    
    # Calculate Specificity using confusion matrix
    # tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(val_labels, val_preds_binary).ravel()
    val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"\nEvaluation Complete!")
    print(f"Total validation samples: {len(val_dataset)}")
    print("-" * 30)
    print(f"AUC:         {val_auc:.4f}")
    print(f"Accuracy:    {val_accuracy:.4f}")
    print(f"F1 Score:    {val_f1:.4f}")
    print(f"Precision:   {val_precision:.4f}")
    print(f"Sensitivity: {val_recall:.4f}  (Recall)")
    print(f"Specificity: {val_specificity:.4f}")
    print("-" * 30)

if __name__ == '__main__':
    main()