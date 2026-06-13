import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

def get_model(num_classes):
    # We must load with weights=None because we are loading our custom weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_npy_slice(npy_path):
    vol = np.load(npy_path)
    if vol.ndim == 4 and vol.shape[1] == 3:
        mid_idx = vol.shape[0] // 2
        img = vol[mid_idx, 0, :, :]
    elif vol.ndim == 3:
        mid_idx = vol.shape[0] // 2
        img = vol[mid_idx, :, :]
    else:
        raise ValueError(f"Unexpected shape {vol.shape}")
        
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img

def main():
    model_path = 'models/joint_detector.pth'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please download it from Sonic HPC first!")
        return

    # Use CPU or Mac Silicon (MPS) for incredibly fast inference
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    
    # Load our trained architecture
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Pick 5 random raw .npy files from the original dataset
    base_dir = "code/src/data/mrnet/train"
    all_files = glob.glob(f"{base_dir}/*/*.npy")
    if not all_files:
        print("Could not find any raw .npy files to test on!")
        return
        
    test_files = random.sample(all_files, 35)

    fig, axes = plt.subplots(5, 7, figsize=(28, 20))
    fig.suptitle("AI Semantic Joint Detection (35 Unseen Raw Images)", fontsize=24)
    axes = axes.flatten()

    print("Running AI inference on 35 random raw .npy files...")
    for i, file_path in enumerate(test_files):
        img_np = load_npy_slice(file_path)
        
        # Convert grayscale to 3-channel RGB to match training data format
        img_rgb = np.stack((img_np,)*3, axis=-1)
        img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)[0]
        
        ax = axes[i]
        ax.imshow(img_np, cmap='gray')
        
        # Draw the bounding box prediction
        if len(prediction['boxes']) > 0:
            box = prediction['boxes'][0].cpu().numpy()
            score = prediction['scores'][0].cpu().item()
            xmin, ymin, xmax, ymax = box
            
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                             linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(f"Confidence: {score*100:.1f}%")
        else:
            ax.set_title("No Joint Detected")
            
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("joint_detection_test.png", bbox_inches='tight')
    print("Done! Open 'joint_detection_test.png' to see the AI's predictions.")

if __name__ == '__main__':
    main()
