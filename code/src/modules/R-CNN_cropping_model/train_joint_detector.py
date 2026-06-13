import os
import glob
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

class JointDataset(Dataset):
    def __init__(self, annotations_csv, images_dir):
        self.images_dir = images_dir
        self.samples = []
        
        # Parse manual annotations
        with open(annotations_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['patient_id']
                plane = row['plane']
                
                # Match to the corresponding image file dynamically since slice num isn't in annotations.csv
                search_path = os.path.join(images_dir, plane, f"{pid}_slice*.png")
                matches = glob.glob(search_path)
                if not matches:
                    continue
                    
                img_path = matches[0]
                
                # Bounding box coordinates
                xmin = float(row['col_min'])
                xmax = float(row['col_max'])
                ymin = float(row['row_min'])
                ymax = float(row['row_max'])
                
                # Failsafe logic to ensure valid boxes
                if xmax <= xmin: xmax = xmin + 1.0
                if ymax <= ymin: ymax = ymin + 1.0
                
                self.samples.append({
                    'img_path': img_path,
                    'boxes': [xmin, ymin, xmax, ymax]
                })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert("RGB")
        
        # Convert PIL image to Torch tensor (C, H, W)
        img_tensor = F.to_tensor(img)
        
        # PyTorch expects boxes in [xmin, ymin, xmax, ymax] format
        boxes = torch.as_tensor([sample['boxes']], dtype=torch.float32)
        
        # Labels: 1 represents our single target class 'Joint Space'
        labels = torch.ones((1,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)
        
        return img_tensor, target

def get_model(num_classes):
    # Load a pre-trained Faster R-CNN with a ResNet50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Get the number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    # num_classes includes background, so it's 2 (Class 0: Background, Class 1: Joint)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='joint_annotations.csv')
    parser.add_argument('--img_dir', type=str, default='annotation_samples')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    dataset = JointDataset(args.csv, args.img_dir)
    print(f"Loaded {len(dataset)} annotated samples.")
    
    # Detection models require a custom collate function to handle variable sized bounding box lists
    def collate_fn(batch):
        return tuple(zip(*batch))
        
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=0, collate_fn=collate_fn
    )
    
    # Check for MPS (Apple Silicon GPU), CUDA, or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")
    
    # Initialize model
    model = get_model(num_classes=2)
    model.to(device)
    
    # Object Detection models often have exploding loss with Adam, so we use standard SGD
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    os.makedirs('models', exist_ok=True)
    
    from tqdm import tqdm
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass generates loss dictionary
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping to prevent exploding loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())
            
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {epoch_loss/len(data_loader):.4f}")
        
    # Save the trained weights
    save_path = os.path.join('models', 'joint_detector.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights securely saved to '{save_path}'")

if __name__ == '__main__':
    main()
