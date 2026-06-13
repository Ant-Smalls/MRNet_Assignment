import argparse
import base64
from io import BytesIO
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import mode
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from modules.data_preprocessing_transformation import MRNetDataset
from modules.baseline_models import create_baseline_model


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM explainability HTML report')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--data_mode', type=str, required=True, choices=['uncropped', 'cropped'])
    parser.add_argument('--output', type=str, default='triage_report.html', help='Output HTML file')
    parser.add_argument('--data_dir', type=str, default='src/data/')
    return parser.parse_args()


def get_gradcam_heatmap(model, exam, target_slice_idx):
    """Generate Grad-CAM heatmap for a specific slice."""
    target_layers = [model.backbone[-2]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Extract single slice and generate heatmap
    slice_input = exam[:, target_slice_idx:target_slice_idx+1, :, :, :]
    grayscale_cam = cam(input_tensor=slice_input)
    
    return grayscale_cam[0]


def generate_triage_html(model, dataset, device, output_path='triage_report.html'):
    """Generate static HTML with top-20 cases showing 4 visualization components."""
    model.eval()
    predictions = []
    
    # Run inference on all validation data
    for idx in range(len(dataset)):
        exam, label = dataset[idx]
        exam_input = exam.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, slice_indices = model.forward_with_slice_tracking(exam_input)
            prob = torch.sigmoid(logits).item()
        
        # Find most informative slice (mode of slice indices)
        most_informative_slice = mode(slice_indices.cpu().numpy(), axis=1, keepdims=False)[0][0]
        contribution_count = (slice_indices == most_informative_slice).sum().item()
        
        predictions.append({
            'idx': idx,
            'true_label': label.item(),
            'pred_prob': prob,
            'slice_idx': most_informative_slice,
            'contribution': contribution_count
        })
    
    # Sort by confidence and select top-20 (10 positive, 10 negative)
    predictions.sort(key=lambda x: x['pred_prob'], reverse=True)
    top_positive = [p for p in predictions if p['pred_prob'] > 0.5][:10]
    top_negative = [p for p in predictions if p['pred_prob'] <= 0.5][-10:]
    top_cases = top_positive + top_negative
    
    # Build HTML
    html = ['<html><head><title>ACL Triage Report</title>']
    html.append('<style>body{font-family:Arial;} .case{margin:30px;border:1px solid #ccc;padding:20px;}</style>')
    html.append('</head><body>')
    html.append('<h1>ACL Detection - Sagittal Plane - Top 20 Cases</h1>')
    
    for case in top_cases:
        exam, label = dataset[case['idx']]
        slice_idx = case['slice_idx']
        
        # Get original slice
        original_slice = exam[slice_idx, 0, :, :].numpy()
        
        # Generate Grad-CAM heatmap
        heatmap = get_gradcam_heatmap(model, exam.unsqueeze(0).to(device), slice_idx)
        
        # Create overlay
        overlay = show_cam_on_image(
            np.stack([original_slice]*3, axis=2), 
            heatmap, 
            use_rgb=True
        )
        
        # Create figure with original and overlay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(original_slice, cmap='gray')
        ax1.set_title('Original Slice')
        ax1.axis('off')
        ax2.imshow(overlay)
        ax2.set_title('Grad-CAM Overlay')
        ax2.axis('off')
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        
        contribution_pct = (case['contribution'] / 512) * 100
        
        # Add case to HTML
        html.append('<div class="case">')
        html.append(f'<h3>Case #{case["idx"]}</h3>')
        html.append(f'<p><strong>Overall Confidence:</strong> {case["pred_prob"]:.3f} ({case["pred_prob"]*100:.1f}%)</p>')
        html.append(f'<p><strong>True Label:</strong> {"Positive (ACL Tear)" if case["true_label"] == 1 else "Negative (No Tear)"}</p>')
        html.append(f'<p><strong>Most Informative Slice:</strong> #{slice_idx}</p>')
        html.append(f'<p><strong>Slice Contribution:</strong> {case["contribution"]}/512 features ({contribution_pct:.1f}%)</p>')
        html.append(f'<img src="data:image/png;base64,{img_b64}" style="width:100%;max-width:800px;"/>')
        html.append('</div>')
    
    html.append('</body></html>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(html))
    
    print(f"Triage report saved to {output_path}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Load checkpoint and extract metadata
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint['args']
    
    print(f"Condition: {checkpoint_args['condition']}, Plane: {checkpoint_args['plane']}")
    print(f"Validation AUC: {checkpoint['val_auc']:.4f}")
    
    # Load model
    model = create_baseline_model().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load validation dataset
    val_dataset = MRNetDataset(
        root_dir=args.data_dir,
        condition=checkpoint_args['condition'],
        plane=checkpoint_args['plane'],
        split='val',
        data_mode=args.data_mode,
        transform=None
    )
    
    print(f"Loaded {len(val_dataset)} validation cases")
    print("Generating Grad-CAM visualizations...")
    
    # Generate HTML report
    generate_triage_html(model, val_dataset, device, args.output)
    
    print(f"\n Open {args.output} in a browser to view the report.")


if __name__ == '__main__':
    main()
