"""
Role 6: Explainability & Clinical Presentation

Grad-CAM visualization on the most important MRI slice.
Shows WHERE the model looked when making its decision.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from modules.data_preprocessing_transformation import MRNetDataset, stack_channels
from modules.baseline_models import create_baseline_model
try:
    from modules.comparative_models import create_comparative_model
except ImportError:
    create_comparative_model = None


# ─────────────────────────────────────────────────────────────
# 1. Load the saved model weights from disk
# ─────────────────────────────────────────────────────────────
def load_model(architecture, checkpoint_path, device):
    """
    Reload a trained model's weights from a saved checkpoint file.
    """
    if architecture == 'baseline':
        model = create_baseline_model()
    else:
        model = create_comparative_model(architecture)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  ✅ Loaded model from {checkpoint_path}")
    return model


# ─────────────────────────────────────────────────────────────
# 2. Tap the wire — register hooks to listen inside the model
# ─────────────────────────────────────────────────────────────
def get_gradcam_hook(model):
    """
    Register forward and backward hooks on the last conv block (layer4).

    Register hook to capture gradients and activations from target layer.
    The forward hook records WHAT the layer saw.
    The backward hook records HOW MUCH each pixel mattered for the final answer.

    Returns:
        (activations_store, gradients_store, [hook1, hook2]) 
    """
    activations_store = {}
    gradients_store   = {}

    # Target layer: last residual block of ResNet18 inside our backbone
    # model.backbone[7] = layer4 of ResNet18
    target_layer = model.backbone[7]

    def forward_hook(module, input, output):
        # Called automatically every time data flows FORWARD through layer4
        activations_store['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        # Called automatically every time gradients flow BACKWARD through layer4
        gradients_store['value'] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    return activations_store, gradients_store, [h1, h2]


# ─────────────────────────────────────────────────────────────
# 3. Compute the Grad-CAM heatmap
# ─────────────────────────────────────────────────────────────
def compute_gradcam(model, exam_tensor, device):
    """
    Compute Grad-CAM heatmap for the whole MRI exam.

    Steps:
    1. Forward pass → get prediction + capture activations
    2. Backward pass → capture which pixels caused the decision
    3. Weight activations by gradient importance
    4. Find the single slice that contributed most (the "key slice")
    5. Return heatmap for that slice + slice index + prediction probability

    exam_tensor: (S, 1, H, W) — the raw exam from the dataset
    """
    # Grad-CAM backward pass may have compatibility issues on MPS.
    # Safest is to compute on CPU for this step.
    model_cpu = model.to('cpu')
    model_cpu.eval()

    # Stack to 3-channel and add batch dim: (1, S, 3, H, W)
    exam_3ch = stack_channels(exam_tensor)         # (S, 3, H, W)
    exam_in  = exam_3ch.unsqueeze(0)               # (1, S, 3, H, W)

    # Register hooks
    activations_store, gradients_store, hooks = get_gradcam_hook(model_cpu)

    # Forward pass (with gradient tracking ON)
    output = model_cpu(exam_in)                    # (1, 1) raw logit
    pred_prob = torch.sigmoid(output).item()       # 0-1 probability

    # Backward pass: compute gradients w.r.t. the prediction score
    model_cpu.zero_grad()
    output.backward()

    # Remove hooks — we have what we need
    for h in hooks:
        h.remove()

    # activations: (S, 512, H', W')  — what layer4 saw for each slice
    # gradients:   (S, 512, H', W')  — how important each neuron was
    acts = activations_store['value']   # (S, 512, H', W')
    grads = gradients_store['value']    # (S, 512, H', W')

    # Global average pool the gradients over the spatial dims (H', W')
    # This gives one importance weight per channel per slice
    weights = grads.mean(dim=(2, 3), keepdim=True)   # (S, 512, 1, 1)

    # Grad-CAM = ReLU( sum over channels of weight * activation )
    # ReLU keeps only what INCREASED the prediction (positive contributions)
    cam = torch.relu((weights * acts).sum(dim=1))    # (S, H', W')

    # Find the "most important slice" — the one with the strongest Grad-CAM signal
    # This is the slice the model stared at hardest
    slice_importance = cam.max(dim=2).values.max(dim=1).values   # (S,)
    key_slice_idx = int(slice_importance.argmax().item())

    # Extract Grad-CAM for the key slice, convert to numpy
    heatmap = cam[key_slice_idx].cpu().numpy()    # (H', W')

    # Put the model back on original device
    model.to(device)

    return heatmap, key_slice_idx, pred_prob


# ─────────────────────────────────────────────────────────────
# 4. Overlay the heatmap onto the MRI image
# ─────────────────────────────────────────────────────────────
def overlay_heatmap(image_slice, heatmap, alpha=0.45):
    """
    Blend the Grad-CAM heatmap on top of the MRI image.

    image_slice: 2D numpy array (H, W) with pixel values 0-255
    heatmap:     2D numpy array — raw Grad-CAM output (any size)
    Returns: RGB overlay image (H, W, 3)
    """
    H, W = image_slice.shape

    # Resize heatmap to match the MRI image dimensions
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Normalise to 0-1  (avoid division by zero)
    hm_min, hm_max = heatmap_resized.min(), heatmap_resized.max()
    if hm_max > hm_min:
        heatmap_norm = (heatmap_resized - hm_min) / (hm_max - hm_min)
    else:
        heatmap_norm = heatmap_resized

    # Convert to colour using the "jet" colormap: blue=ignored, red=stared here
    colormap = cm.jet(heatmap_norm)[:, :, :3]     # (H, W, 3) RGB, 0-1 range

    # Normalise the original MRI to 0-1
    mri_rgb = np.stack([image_slice / 255.0] * 3, axis=-1)   # (H, W, 3)

    # Blend: 55% MRI + 45% heatmap colour
    overlay = (1 - alpha) * mri_rgb + alpha * colormap
    overlay = np.clip(overlay, 0, 1)
    return overlay, heatmap_norm


# ─────────────────────────────────────────────────────────────
# 5. Draw the final clinical visualization
# ─────────────────────────────────────────────────────────────
def generate_visualization(exam_tensor, heatmap, slice_idx,
                           pred_prob, label, condition, save_path):
    """
    Create a 3-panel image:
      Panel 1: Original MRI slice (greyscale)
      Panel 2: Grad-CAM heatmap alone
      Panel 3: MRI + heatmap blended (the money shot)

    Generates the Grad-CAM visualization for clinical review, overlaying
    heatmap activations to verify anatomical focus.
    """
    # Get raw pixel values of the key slice (0-255 uint8)
    # exam_tensor shape: (S, 1, H, W) — we need (H, W) for the key slice
    image_slice = (exam_tensor[slice_idx, 0].cpu().numpy() * 255).astype(np.uint8)

    overlay, heatmap_norm = overlay_heatmap(image_slice, heatmap)

    # Clinical labels for the title
    pred_label = "TEAR DETECTED" if pred_prob >= 0.5 else "No Tear"
    true_label = "ACL Tear" if label == 1 else "Normal"
    colour = "#ff4444" if pred_prob >= 0.5 else "#44ff88"
    correct = "✓ CORRECT" if (pred_prob >= 0.5) == (label == 1) else "✗ WRONG"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        f"{condition.upper()} | Ground Truth: {true_label}  |  "
        f"Model says: {pred_label} ({pred_prob:.1%})  {correct}",
        color=colour, fontsize=13, fontweight='bold', y=1.01
    )

    titles = ["Original MRI Slice", "Grad-CAM Heatmap", "Overlay (Where it looked)"]
    images = [image_slice, heatmap_norm, overlay]
    cmaps  = ["gray",     "jet",         None]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.set_facecolor("#0d0d0d")
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_xlabel(f"Slice {slice_idx}", color="gray", fontsize=8)
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(1.5)

    # Add a colour bar showing the heatmap scale
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.04, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='gray', labelsize=7)
    cbar.set_label("Attention Intensity\n(red = model stared here)",
                   color='white', fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"  🔥 Saved Grad-CAM -> {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────
# 6. Orchestrate: pick interesting cases & visualise them
# ─────────────────────────────────────────────────────────────
def run_gradcam_pipeline(condition, plane, architecture,
                         checkpoint_path, data_dir, output_dir,
                         num_examples=3):
    """
    End-to-end:
    1. Load model
    2. Run all validation exams
    3. Pick the most interesting cases (confident correct, errors, borderline)
    4. Generate Grad-CAM images for each
    """
    device = _get_device()

    model = load_model(architecture, checkpoint_path, device)
    val_dataset = MRNetDataset(data_dir, condition, plane, split='valid')
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Collect predictions for all validation exams
    print(f"\n  Running inference on {len(val_dataset)} validation exams...")
    results = []
    with torch.no_grad():
        for idx, (exam, label) in enumerate(val_loader):
            exam_sq = exam.squeeze(0)                  # (S, 1, H, W)
            exam_3ch = stack_channels(exam_sq)
            exam_in  = exam_3ch.unsqueeze(0).to(device)
            output   = model(exam_in)
            prob     = torch.sigmoid(output).item()
            results.append({
                'idx': idx,
                'prob': prob,
                'label': label.item(),
                'exam': exam_sq,          # Keep raw (S,1,H,W) for later
            })

    # Pick interesting cases
    # - High confidence CORRECT: model was very sure and right
    # - Errors (FP or FN): model was wrong — crucial for clinical review
    # - Borderline: near the 0.5 decision boundary
    tp = [r for r in results if r['prob'] >= 0.7 and r['label'] == 1]   # confident true positive
    tn = [r for r in results if r['prob'] <= 0.3 and r['label'] == 0]   # confident true negative
    fn = [r for r in results if r['prob'] < 0.5 and r['label'] == 1]    # missed tear!
    fp = [r for r in results if r['prob'] >= 0.5 and r['label'] == 0]   # false alarm
    bl = sorted(results, key=lambda r: abs(r['prob'] - 0.5))[:num_examples]  # borderline

    cases = {}
    if tp: cases['confident_correct_tear']    = tp[:num_examples]
    if tn: cases['confident_correct_normal']  = tn[:num_examples]
    if fn: cases['missed_tear_FN']            = fn[:num_examples]
    if fp: cases['false_alarm_FP']            = fp[:num_examples]
    if bl: cases['borderline']                = bl[:num_examples]

    print(f"\n  Found: {len(tp)} confident TPs | {len(tn)} confident TNs | "
          f"{len(fn)} missed tears (FN) | {len(fp)} false alarms (FP)")

    # Generate Grad-CAM for each picked case
    saved_paths = []
    for category, case_list in cases.items():
        for i, case in enumerate(case_list):
            print(f"\n  Computing Grad-CAM: {category} case {i+1}...")
            heatmap, key_slice, pred_prob = compute_gradcam(
                model, case['exam'], device)

            save_path = os.path.join(output_dir,
                                     f"{condition}_{plane}",
                                     f"{category}_{i+1}.png")
            generate_visualization(
                exam_tensor=case['exam'],
                heatmap=heatmap,
                slice_idx=key_slice,
                pred_prob=pred_prob,
                label=case['label'],
                condition=condition,
                save_path=save_path
            )
            saved_paths.append(save_path)

    print(f"\n  ✅ Generated {len(saved_paths)} Grad-CAM visualisations in {output_dir}")
    return saved_paths


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition',       type=str, default='acl')
    parser.add_argument('--plane',           type=str, default='sagittal')
    parser.add_argument('--architecture',    type=str, default='baseline')
    parser.add_argument('--checkpoint_path', type=str,
                        default='local_learning_hub/checkpoints/acl_sagittal_baseline/best_phase2.pth')
    parser.add_argument('--data_dir',        type=str, default='code/src/data/mrnet/')
    parser.add_argument('--output_dir',      type=str, default='local_learning_hub/visualisations/')
    parser.add_argument('--num_examples',    type=int, default=2)
    args = parser.parse_args()

    run_gradcam_pipeline(
        condition=args.condition,
        plane=args.plane,
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_examples=args.num_examples
    )
