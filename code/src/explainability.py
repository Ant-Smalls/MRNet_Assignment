"""
Explainability: Grad-CAM heatmap generation for MRNet AlexNet models.

Runs Grad-CAM on the most informative slice of each validation patient,
across all conditions and planes, and saves results to a JSON file
that the triage dashboard can embed.

Usage (runs all 18 checkpoints automatically):
    python code/src/explainability.py \
        --checkpoint_dir checkpoints/ \
        --data_dir code/src/data/ \
        --output_dir job_outputs/gradcam/ \
        --data_mode cropped \
        --n_cases 10

Or for a single checkpoint:
    python code/src/explainability.py \
        --checkpoint checkpoints/acl_axial_baseline_cropped/best_single_phase.pth \
        --data_dir code/src/data/ \
        --data_mode cropped
"""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mode
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from modules.data_preprocessing_transformation import ValidMRNetDataset
from modules.baseline_models import create_baseline_model
from modules.comparative_models import create_comparative_model


# ── GradCAM wrapper ─────────────────────────────────────────────────────────

class SliceWrapper(nn.Module):
    """Wraps MRNetBaseModel to accept a single 4D slice for GradCAM compatibility.
    GradCAM provides (B, 3, H, W); the base model expects (B, S, 3, H, W)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.backbone = model.backbone
        self.fc = model.fc

    def forward(self, x):
        return self.model(x.unsqueeze(1))


def gradcam_for_slice(model, slice_tensor, device):
    """Return a (H, W) Grad-CAM heatmap for one slice.

    AlexNet features layer indices:
      0  Conv2d(3,64,11,4,2)
      1  ReLU
      2  MaxPool2d
      3  Conv2d(64,192,5,1,2)
      4  ReLU
      5  MaxPool2d
      6  Conv2d(192,384,3,1,1)
      7  ReLU
      8  Conv2d(384,256,3,1,1)
      9  ReLU
      10 Conv2d(256,256,3,1,1)   <- target: last conv before final relu
      11 ReLU
      12 MaxPool2d
    """
    wrapped = SliceWrapper(model).to(device)
    
    if hasattr(wrapped.backbone, 'features'):
        # DenseNet (XRVBackbone)
        target_layer = [wrapped.backbone.features[-1]]
    else:
        # AlexNet (Sequential)
        target_layer = [wrapped.backbone[10]]
        
    cam = GradCAM(model=wrapped, target_layers=target_layer)
    grayscale = cam(input_tensor=slice_tensor.unsqueeze(0).to(device))
    return grayscale[0]


# ── Inference helpers ────────────────────────────────────────────────────────

def most_informative_slice(model, exam_tensor, device):
    """Return the slice index that contributed the most features via max-pool."""
    with torch.no_grad():
        logits, slice_indices, pooled = model.forward_with_slice_tracking(
            exam_tensor.unsqueeze(0).to(device)
        )
    # Ignore channels where the max activation was 0 (dead neurons)
    # Otherwise torch.max defaults to slice 0 for all zero-channels, skewing the mode!
    pooled = pooled.cpu().numpy().flatten()
    idxs = slice_indices.cpu().numpy().flatten()
    active_idxs = idxs[pooled > 0]
    
    if len(active_idxs) == 0:
        return int(mode(idxs, keepdims=False)[0])  # Fallback
    return int(mode(active_idxs, keepdims=False)[0])


def run_inference(model, dataset, device):
    """Return list of dicts with pred_prob, true_label, slice_idx per patient."""
    results = []
    for i in tqdm(range(len(dataset)), desc="    Inference", leave=False):
        exam, label = dataset[i]
        exam_t = exam.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(exam_t)
            prob = torch.sigmoid(logits).item()
        s_idx = most_informative_slice(model, exam, device)
        results.append({
            'idx': i,
            'true_label': int(label.item()),
            'pred_prob': float(prob),
            'slice_idx': s_idx,
        })
    return results


# ── Visualisation ────────────────────────────────────────────────────────────

def normalise(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def figure_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_gradcam_figure(exam, slice_idx, heatmap, pred_prob, true_label,
                         condition, plane, patient_id):
    """Create a side-by-side original / Grad-CAM overlay figure."""
    raw_slice = exam[slice_idx, 0, :, :].numpy()
    rgb = np.stack([normalise(raw_slice)] * 3, axis=2).astype(np.float32)

    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True, image_weight=0.55)

    outcome = _outcome(pred_prob, true_label)
    outcome_color = {'TP': '#059669', 'TN': '#0891b2',
                     'FP': '#dc2626', 'FN': '#d97706'}[outcome]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.patch.set_facecolor('#f8fafc')

    for ax, img, title, cmap in zip(
        axes,
        [raw_slice, overlay],
        [f'Original MRI — Slice #{slice_idx}', 'Grad-CAM Attention Overlay'],
        ['gray', None]
    ):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight='600', pad=6, color='#1e293b')
        ax.axis('off')

    label_str = f"{'Positive' if true_label else 'Negative'} · "
    pred_str  = f"Pred: {pred_prob*100:.1f}% · Outcome: {outcome}"
    fig.suptitle(
        f"Patient {patient_id} — {condition.upper()} {plane.capitalize()}\n"
        f"{label_str}{pred_str}",
        fontsize=9.5, color='#334155', y=1.01
    )
    # Colour the outcome label
    fig.text(0.5, -0.01,
             f'■ {outcome}',
             ha='center', fontsize=9, fontweight='700',
             color=outcome_color, transform=fig.transFigure)
    fig.tight_layout()
    return fig


def _outcome(prob, true_label, thr=0.5):
    pred = prob >= thr
    real = true_label == 1
    if pred and real:     return 'TP'
    if not pred and not real: return 'TN'
    if pred and not real: return 'FP'
    return 'FN'


# ── Per-checkpoint pipeline ──────────────────────────────────────────────────

def process_checkpoint(ckpt_dir, data_dir, data_mode, n_cases, device):
    """Run Grad-CAM for one condition/plane checkpoint. Returns a result dict."""
    ckpt_path = _find_checkpoint(ckpt_dir)
    if ckpt_path is None:
        return None

    config_path = Path(ckpt_dir) / 'config.json'
    if not config_path.exists():
        print(f"  [skip] no config.json in {ckpt_dir}")
        return None

    with open(config_path) as f:
        cfg = json.load(f)

    condition = cfg['condition']
    plane     = cfg['plane']
    ckpt_data_mode = cfg.get('data_mode', data_mode)
    architecture = cfg.get('architecture', 'baseline')
    comparative_arch = cfg.get('comparative_arch', 'xrv_dense')

    if ckpt_data_mode != data_mode:
        return None

    print(f"  Processing: {condition} / {plane} / {architecture} / {ckpt_data_mode}")

    if architecture == 'baseline':
        model = create_baseline_model().to(device)
    else:
        model = create_comparative_model(architecture=comparative_arch).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    dataset = ValidMRNetDataset(
        root_dir=data_dir,
        condition=condition,
        plane=plane,
        data_mode=data_mode,
    )

    if len(dataset) == 0:
        print(f"  [warn] empty dataset for {condition}/{plane}/{data_mode}")
        return None

    print(f"    Loaded {len(dataset)} patients, running inference…")
    preds = run_inference(model, dataset, device)

    # Select n_cases evenly spaced across the patient list.
    # Using patient-index order (not outcome order) ensures that BOTH the
    # cropped and uncropped models select the EXACT SAME patients,
    # which prevents "Patient not in selection" in the dashboard comparison.
    preds_by_idx = sorted(preds, key=lambda p: p['idx'])
    n_total = len(preds_by_idx)
    if n_total <= n_cases:
        selected = preds_by_idx
    else:
        step = n_total / n_cases
        selected = [preds_by_idx[int(i * step)] for i in range(n_cases)]

    print(f"    Generating Grad-CAM for {len(selected)} cases…")
    cases_out = []
    for case in tqdm(selected, desc=f"  {condition}-{plane}", leave=False):
        exam, _ = dataset[case['idx']]
        s_idx   = case['slice_idx']
        heatmap = gradcam_for_slice(
            model,
            exam[s_idx],        # (3, H, W)
            device
        )
        fig = make_gradcam_figure(
            exam=exam,
            slice_idx=s_idx,
            heatmap=heatmap,
            pred_prob=case['pred_prob'],
            true_label=case['true_label'],
            condition=condition,
            plane=plane,
            patient_id=dataset.file_paths[case['idx']].split('/')[-1].replace('.npy', '')
        )
        img_b64 = figure_to_b64(fig)
        cases_out.append({
            **case,
            'outcome': _outcome(case['pred_prob'], case['true_label']),
            'img_b64': img_b64,
        })

    return {
        'condition': condition,
        'plane': plane,
        'data_mode': data_mode,
        'architecture': architecture,
        'comparative_arch': comparative_arch if architecture != 'baseline' else None,
        'val_auc': float(ckpt.get('val_auc', 0)),
        'n_total': len(dataset),
        'cases': cases_out,
    }


def _find_checkpoint(model_dir):
    for name in ['best_phase2.pth', 'best_single_phase.pth', 'best_phase1.pth']:
        p = Path(model_dir) / name
        if p.exists():
            return p
    hits = list(Path(model_dir).glob('best_*.pth'))
    return hits[0] if hits else None


# ── Entry points ─────────────────────────────────────────────────────────────

def run_all(checkpoint_dir, data_dir, output_dir, data_mode, n_cases, device):
    """Iterate every checkpoint subdirectory and collect Grad-CAM results."""
    ckpt_root = Path(checkpoint_dir)
    out_root  = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ckpt_dir in sorted(ckpt_root.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        result = process_checkpoint(ckpt_dir, data_dir, data_mode, n_cases, device)
        if result:
            all_results.append(result)

    out_path = out_root / f'gradcam_{data_mode}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f)
    print(f"\nGrad-CAM results saved → {out_path}")
    return all_results


def run_single(ckpt_path, data_dir, data_mode, n_cases, device):
    """Run Grad-CAM for a single checkpoint (reads config.json from same dir)."""
    ckpt_dir = Path(ckpt_path).parent
    result = process_checkpoint(ckpt_dir, data_dir, data_mode, n_cases, device)
    if result:
        print(f"\nDone: {result['condition']} / {result['plane']} / {result['data_mode']}")
        print(f"  Cases: {len(result['cases'])} | Val AUC: {result['val_auc']:.4f}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',     type=str, default=None,
                   help='Single checkpoint .pth file (omit to run all)')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                   help='Directory of all checkpoints (used when --checkpoint is omitted)')
    p.add_argument('--data_dir',       type=str, default='code/src/data/')
    p.add_argument('--output_dir',     type=str, default='job_outputs/gradcam/')
    p.add_argument('--data_mode',      type=str, default='cropped',
                   choices=['cropped', 'uncropped'])
    p.add_argument('--n_cases',        type=int, default=10,
                   help='Number of cases to visualise per condition/plane')
    return p.parse_args()


def main():
    args   = parse_args()
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"Device: {device} | Data mode: {args.data_mode} | Cases per model: {args.n_cases}")

    if args.checkpoint:
        run_single(args.checkpoint, args.data_dir, args.data_mode, args.n_cases, device)
    else:
        run_all(args.checkpoint_dir, args.data_dir, args.output_dir,
                args.data_mode, args.n_cases, device)


if __name__ == '__main__':
    main()
