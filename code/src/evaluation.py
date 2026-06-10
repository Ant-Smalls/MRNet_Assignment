"""
Role 5: Evaluation & Metrics

Comprehensive evaluation with multi-plane fusion, metrics, and visualization.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib
matplotlib.use("Agg")   # Save to file (no screen needed)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from modules.data_preprocessing_transformation import MRNetDataset, stack_channels
from modules.baseline_models import create_baseline_model
try:
    from modules.comparative_models import create_comparative_model
except ImportError:
    create_comparative_model = None


# ─────────────────────────────────────────────
# 1.  Load the saved model brains from disk
# ─────────────────────────────────────────────
def load_models(condition, architecture, checkpoint_dir):
    """
    Load the best saved checkpoint for every plane that exists.

    Returns:
        dict: {'axial': model, 'coronal': model, 'sagittal': model}
              Missing planes are simply omitted from the dict.
    """
    device = _get_device()
    models = {}

    for plane in ['axial', 'coronal', 'sagittal']:
        # e.g.  checkpoints/acl_sagittal_baseline/best_phase2.pth
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"{condition}_{plane}_{architecture}",
            "best_phase2.pth"
        )
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  No checkpoint for {plane} plane — skipping.")
            continue

        # Re-create the exact same model architecture
        if architecture == 'baseline':
            model = create_baseline_model()
        else:
            model = create_comparative_model(architecture)

        # Load the saved brain weights back in
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()   # Test mode — no learning allowed
        models[plane] = model
        print(f"  ✅ Loaded {plane} model (saved at epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")

    return models


# ─────────────────────────────────────────────
# 2.  Run one model over all validation exams
# ─────────────────────────────────────────────
def get_plane_predictions(model, dataloader, device):
    """
    Feed all validation exams through one plane model.
    Returns: (predictions, labels) as numpy arrays
    """
    all_probs, all_labels = [], []

    with torch.no_grad():
        for exam, label in dataloader:
            exam = stack_channels(exam.squeeze(0))   # (S,1,H,W) -> (S,3,H,W)
            exam = exam.unsqueeze(0).to(device)       # (1,S,3,H,W)

            output = model(exam)                      # Raw logit (1,1)
            prob = torch.sigmoid(output).item()       # Convert to 0-1 probability
            all_probs.append(prob)
            all_labels.append(label.item())

    return np.array(all_probs), np.array(all_labels)


# ─────────────────────────────────────────────
# 3.  Calculate metrics + confidence intervals
# ─────────────────────────────────────────────
def compute_metrics_with_ci(labels, predictions, threshold=0.5, n_bootstraps=1000):
    """
    Compute AUC, Sensitivity, Specificity, F1 with 95% Confidence Intervals.

    How bootstrap CI works (like a clinical trial):
      - Randomly resample the 120 patients 1,000 times (with replacement)
      - Calculate AUC for each resample
      - The middle 95% of those 1,000 AUC values = your 95% CI
    """
    binary_preds = (predictions >= threshold).astype(int)

    # Base metrics on the actual 120 validation patients
    base_auc         = roc_auc_score(labels, predictions)
    base_sensitivity = recall_score(labels, binary_preds, zero_division=0)  # True Positive Rate
    base_specificity = recall_score(1 - labels, 1 - binary_preds, zero_division=0)
    base_f1          = f1_score(labels, binary_preds, zero_division=0)

    # Bootstrap to get confidence intervals
    rng = np.random.RandomState(42)   # Fixed seed for reproducibility
    boot_aucs, boot_sens, boot_spec, boot_f1s = [], [], [], []

    for _ in range(n_bootstraps):
        # Randomly pick 120 patients (some may be picked more than once)
        indices = rng.randint(0, len(labels), len(labels))
        s_labels = labels[indices]
        s_preds  = predictions[indices]
        s_binary = binary_preds[indices]

        if len(np.unique(s_labels)) < 2:
            continue  # Skip if resample has only one class (can't compute AUC)

        boot_aucs.append(roc_auc_score(s_labels, s_preds))
        boot_sens.append(recall_score(s_labels, s_binary, zero_division=0))
        boot_spec.append(recall_score(1 - s_labels, 1 - s_binary, zero_division=0))
        boot_f1s.append(f1_score(s_labels, s_binary, zero_division=0))

    def ci(values, base):
        """Return (value, lower 95% CI, upper 95% CI)"""
        return base, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    return {
        'auc':         ci(boot_aucs, base_auc),
        'sensitivity': ci(boot_sens, base_sensitivity),   # Most important clinical metric!
        'specificity': ci(boot_spec, base_specificity),
        'f1':          ci(boot_f1s,  base_f1),
        'threshold':   threshold,
        'n_patients':  len(labels),
    }


# ─────────────────────────────────────────────
# 4.  Draw the ROC and Precision-Recall curves
# ─────────────────────────────────────────────
def plot_curves(labels, predictions, title, save_dir):
    """
    Plot ROC and PR curves side by side.
    The ROC curve is the graph your professor will almost certainly want to see.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d0d0d")

    auc_score = roc_auc_score(labels, predictions)
    avg_prec  = average_precision_score(labels, predictions)

    # ── Left panel: ROC Curve ──
    fpr, tpr, _ = roc_curve(labels, predictions)
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.plot(fpr, tpr, color="#7b2ff7", linewidth=2.5, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], color="#555", linestyle="--", linewidth=1, label="Random (AUC=0.5)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#7b2ff7")
    ax.set_xlabel("False Positive Rate\n(Healthy knees incorrectly flagged)", color="white")
    ax.set_ylabel("True Positive Rate\n(Actual tears correctly caught)", color="white")
    ax.set_title(f"ROC Curve — {title}", color="white", fontweight="bold")
    ax.legend(loc="lower right", facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")

    # ── Right panel: Precision-Recall Curve ──
    prec, rec, _ = precision_recall_curve(labels, predictions)
    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    ax.plot(rec, prec, color="#00d4ff", linewidth=2.5, label=f"Avg Precision = {avg_prec:.3f}")
    ax.axhline(y=labels.mean(), color="#555", linestyle="--", linewidth=1,
               label=f"Random (prevalence={labels.mean():.2f})")
    ax.fill_between(rec, prec, alpha=0.15, color="#00d4ff")
    ax.set_xlabel("Recall (Sensitivity)", color="white")
    ax.set_ylabel("Precision (Positive Predictive Value)", color="white")
    ax.set_title(f"Precision-Recall Curve — {title}", color="white", fontweight="bold")
    ax.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  📊 Saved curves -> {save_path}")
    return save_path


# ─────────────────────────────────────────────
# 5.  Orchestrate everything end-to-end
# ─────────────────────────────────────────────
def evaluate_architecture(condition, architecture, checkpoint_dir, data_dir, output_dir):
    """
    Complete evaluation pipeline:
    1. Load all available plane models
    2. Get each plane's predictions on the validation set
    3. Fuse predictions (average the 3 planes)
    4. Compute metrics with confidence intervals
    5. Plot and save the ROC/PR curves
    """
    device = _get_device()
    os.makedirs(output_dir, exist_ok=True)

    # Load whichever plane models we have saved checkpoints for
    models = load_models(condition, architecture, checkpoint_dir)
    if not models:
        print("❌ No trained models found. Please run train.py first.")
        return

    # Per-plane predictions
    all_preds = {}
    all_labels = None

    for plane, model in models.items():
        print(f"\n  Running inference on {plane} validation set...")
        val_dataset = MRNetDataset(data_dir, condition, plane, split='valid')
        val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False)

        preds, labels = get_plane_predictions(model, val_loader, device)
        all_preds[plane] = preds
        all_labels = labels   # Labels are the same for every plane

    # Fuse predictions: average across all available planes
    fused_preds = np.mean(list(all_preds.values()), axis=0)

    # Print per-plane metrics
    print(f"\n{'='*60}")
    print(f"  Results for condition: {condition.upper()} | {architecture}")
    print(f"{'='*60}")
    for plane, preds in all_preds.items():
        try:
            auc = roc_auc_score(all_labels, preds)
            print(f"  {plane.capitalize():10s} AUC: {auc:.4f}")
        except Exception:
            print(f"  {plane.capitalize():10s} AUC: N/A")

    # Compute and print fused metrics with confidence intervals
    metrics = compute_metrics_with_ci(all_labels, fused_preds)
    auc_v, auc_lo, auc_hi     = metrics['auc']
    sens_v, sens_lo, sens_hi  = metrics['sensitivity']
    spec_v, spec_lo, spec_hi  = metrics['specificity']
    f1_v, f1_lo, f1_hi        = metrics['f1']

    print(f"\n  FUSED (all planes averaged):")
    print(f"  {'AUC':15s} {auc_v:.4f}  (95% CI: {auc_lo:.3f} – {auc_hi:.3f})")
    print(f"  {'Sensitivity':15s} {sens_v:.4f}  (95% CI: {sens_lo:.3f} – {sens_hi:.3f})  ← most clinically important")
    print(f"  {'Specificity':15s} {spec_v:.4f}  (95% CI: {spec_lo:.3f} – {spec_hi:.3f})")
    print(f"  {'F1 Score':15s} {f1_v:.4f}  (95% CI: {f1_lo:.3f} – {f1_hi:.3f})")
    print(f"  Threshold used: {metrics['threshold']} | Patients evaluated: {metrics['n_patients']}")
    print(f"{'='*60}")

    # Plot the ROC and PR curves and save as an image
    curves_path = plot_curves(all_labels, fused_preds,
                              title=f"{condition.upper()} {architecture}",
                              save_dir=output_dir)
    return metrics, curves_path


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition',       type=str, default='acl')
    parser.add_argument('--architecture',    type=str, default='baseline')
    parser.add_argument('--checkpoint_dir',  type=str, default='local_learning_hub/checkpoints/')
    parser.add_argument('--data_dir',        type=str, default='code/src/data/mrnet/')
    parser.add_argument('--output_dir',      type=str, default='local_learning_hub/results/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nEvaluating {args.architecture} for {args.condition.upper()}...\n")
    evaluate_architecture(args.condition, args.architecture,
                          args.checkpoint_dir, args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
