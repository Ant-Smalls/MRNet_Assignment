import os
import glob
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def process_volume(vol_path, model, device, output_path):
    vol = np.load(vol_path)
    
    # Grab middle slice to find the joint
    if vol.ndim == 4 and vol.shape[1] == 3:
        mid_idx = vol.shape[0] // 2
        img = vol[mid_idx, 0, :, :]
    elif vol.ndim == 3:
        mid_idx = vol.shape[0] // 2
        img = vol[mid_idx, :, :]
    else:
        print(f"Skipping {vol_path}: Unknown shape {vol.shape}")
        return False
        
    # Normalize for the model
    img_norm = img - np.min(img)
    img_norm = img_norm / (np.max(img_norm) + 1e-8)
    img_norm = (img_norm * 255).astype(np.uint8)
    
    img_rgb = np.stack((img_norm,)*3, axis=-1)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    # Predict Joint
    with torch.no_grad():
        prediction = model(img_tensor)[0]
    
    if len(prediction['boxes']) > 0:
        box = prediction['boxes'][0].cpu().numpy()
        xmin, ymin, xmax, ymax = map(int, box)
        
        # Add 15px safety padding
        pad = 15
        ymin = max(0, ymin - pad)
        ymax = min(img.shape[0], ymax + pad)
        xmin = max(0, xmin - pad)
        xmax = min(img.shape[1], xmax + pad)
        
        # Crop the entire 3D volume
        if vol.ndim == 4:
            cropped_vol = vol[:, :, ymin:ymax, xmin:xmax]
        else:
            cropped_vol = vol[:, ymin:ymax, xmin:xmax]
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, cropped_vol)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Auto-crop MRNet dataset using Faster R-CNN")
    parser.add_argument('--model_path', type=str, default='models/joint_detector.pth')
    parser.add_argument('--input_dir', type=str, default='data/mrnet/')
    parser.add_argument('--output_dir', type=str, default='data/mrnet_cropped/')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Could not find model at {args.model_path}")
        return

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_files = glob.glob(os.path.join(args.input_dir, "**/*.npy"), recursive=True)
    print(f"Found {len(all_files)} volumes to crop.")

    success_count = 0
    for f in tqdm(all_files):
        rel_path = os.path.relpath(f, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        if process_volume(f, model, device, out_path):
            success_count += 1
            
    print(f"Finished! Successfully cropped {success_count}/{len(all_files)} volumes.")

if __name__ == '__main__':
    main()
