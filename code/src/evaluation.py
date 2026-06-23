"""
Role 5: Evaluation & Metrics

* Multi-plane fusion
* bootstrapped metrics
* ROC/PR curves 
* 2x2 comparison matrix
* prediction export for the Role 6 triage interface.

Implemenation notes:
  - Checkpoints live in checkpoints/{condition}_{plane}_{arch}_{mode}/best_*.pth
  - Metadata is read from that folder's config.json (NOT from inside the .pth)
  - The held-out test set is loaded from ValidMRNetDataset (mrnet/valid/) (this is what my folder setup looks like anyways)
"""

import argparse
#import os
#import glob
from pathlib import Path
import json
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    recall_score, f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.data_preprocessing_transformation import ValidMRNetDataset, MRNetDataset
from modules.baseline_models import create_baseline_model
from modules.comparative_models import create_comparative_model

PLANES = ['axial', 'coronal', 'sagittal']
ARCH_LABELS = {'baseline': 'AlexNet-ImageNet', 'comparative': 'ResNet50-RadImageNet'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True,
                        choices=['acl', 'meniscus', 'abnormal', 'all']) # 'all' added for master table
    parser.add_argument('--architectures', type=str, nargs='+', default=['baseline', 'comparative'],
                        choices=['baseline', 'comparative'])
    parser.add_argument('--data_modes', type=str, nargs='+', default=['uncropped', 'cropped'],
                        choices=['uncropped', 'cropped'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--data_dir', type=str, default='src/data/')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--n_bootstraps', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--tune_threshold', action='store_true',
                        help='Report metrics at a Youden-J threshold tuned on the validation split') # to tune operating threshold
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


########################################
# Finding and loading checkpoints 


def find_checkpoint(model_dir):
    """Return the path of the final best checkpoint, preferring phase2 > single_phase > phase1."""
    for name in ['best_phase2.pth', 'best_single_phase.pth', 'best_phase1.pth']:
        path = Path(model_dir) / name
        if path.exists():
            return path
    hits = list(Path(model_dir).glob('best_*.pth'))
    return hits[0] if hits else None


def load_single_model(condition, plane, architecture, data_mode, checkpoint_dir, device):
    """Load one trained plane model, or return None if its checkpoint isn't available yet."""
    model_name = f"{condition}_{plane}_{architecture}_{data_mode}"
    model_dir = Path(checkpoint_dir) / model_name
    ckpt_path = find_checkpoint(model_dir)
    if ckpt_path is None:
        print(f"  [skip] no checkpoint found in {model_dir}")
        return None

    if architecture == 'baseline':
        model = create_baseline_model()
    else:
        model = create_comparative_model()
        if model is None:
            print(f"  [skip] create_comparative_model() returned None (Role 3 not implemented)")
            return None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint # to accpet either a full checkpoint dict or an empty state_dict
    model.load_state_dict(state)
    model.to(device).eval()
    val_auc = checkpoint.get('val_auc', float('nan'))
    print(f"  [load] {model_name}  (val_auc={val_auc:.4f})  <- {Path(ckpt_path).name}")
    return model


###############################################
# Performing inference (per-plane, per-patient)


def patient_ids_from_dataset(dataset):
    """Parse integer patient IDs from the dataset's file paths ('0065.npy' goes to 65)."""
    return [int(Path(p).stem) for p in dataset.file_paths]


def get_plane_predictions(model, dataset, device):
    """Run inference over the test set. Returns (probs, labels, patient_ids) as aligned arrays.
    shuffle=False keeps the loader order identical to dataset.file_paths, so probs[i]
    lines up with patient_ids[i]."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for exam, label in loader:
            exam = exam.to(device)
            logits = model(exam).squeeze() #squeezing the (1,1) output to a scalar prob
            prob = torch.sigmoid(logits).item()
            probs.append(prob)
            labels.append(int(label.item()))
    return np.array(probs), np.array(labels), np.array(patient_ids_from_dataset(dataset))


####################################
# Defining the evaluation metrics


def _point_metrics(labels, probs, threshold):
    """AUC, sensitivity, specificity, F1 at a fixed threshold."""
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float('nan')
    sens = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel() # 'labels=[0,1]' makes sure that a 2x2 matrix is returned even if one class is missing from bootstrap resampling
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {'auc': auc, 'sensitivity': sens, 'specificity': spec, 'f1': f1}


def compute_metrics_with_ci(labels, predictions, threshold=0.5, n_bootstraps=1000, seed=42):
    """Point metrics plus 95% bootstrap Confidence Intervals. 
        It Returns {metric: (value, ci_low, ci_high)}."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    point = _point_metrics(labels, predictions, threshold)

    rng = np.random.default_rng(seed)
    n = len(labels)
    samples = {k: [] for k in point}
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if len(set(labels[idx])) < 2: #skips the reasmpls that only have one class (auc not defined)
            continue
        m = _point_metrics(labels[idx], predictions[idx], threshold)
        for k, v in m.items():
            if not np.isnan(v):
                samples[k].append(v)

    out = {}
    for k, v in point.items():
        if samples[k]:
            lo, hi = np.percentile(samples[k], [2.5, 97.5])
        else:
            lo, hi = float('nan'), float('nan')
        out[k] = (v, lo, hi)
    return out

# Youden-J Threshold helper
def optimal_threshold(labels, probs):
    """Threshold maximising Youden's J (sensitivity + specificity - 1) on the given set.
    Intended to be called on VAL predictions, then applied to the test set."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return float(thresholds[np.argmax(tpr - fpr)])

#Bootstrapping AUC difference for paired samples
def paired_bootstrap_auc_diff(labels, probs_a, probs_b, n_bootstraps=1000, seed=42):
    """Paired bootstrap on AUC(b) - AUC(a) over the SAME patients.
    Returns (diff, ci_low, ci_high, p_two_sided). CI excluding 0 => significant difference."""
    labels = np.asarray(labels)
    probs_a, probs_b = np.asarray(probs_a), np.asarray(probs_b)
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if len(set(labels[idx])) < 2:
            continue
        diffs.append(roc_auc_score(labels[idx], probs_b[idx]) - roc_auc_score(labels[idx], probs_a[idx]))
    diffs = np.array(diffs)
    point = roc_auc_score(labels, probs_b) - roc_auc_score(labels, probs_a)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) #two-sided bootstrap p-value
    return point, lo, hi, p


#################################################################
# Fusing (so that Pat ID's are aligned and not index alligned) 


def fuse_planes(plane_results):
    """Average per-plane probabilities for patients present in ALL available planes.
    plane_results: {plane: (probs, labels, pids)}.
    Returns (fused_probs, labels, pids) aligned by patient id."""
    label_by_pid = {}
    prob_by_pid = {}
    for plane, (probs, labels, pids) in plane_results.items():
        for prob, lab, pid in zip(probs, labels, pids):
            prob_by_pid.setdefault(pid, {})[plane] = prob
            label_by_pid[pid] = lab

    n_planes = len(plane_results)
    common = sorted(pid for pid, d in prob_by_pid.items() if len(d) == n_planes) # only keep pats that have a pred from every plane
    if len(common) < len(label_by_pid):
        print(f"  [fuse] {len(common)} patients common to all {n_planes} planes "
              f"(dropped {len(label_by_pid) - len(common)} with missing planes)")

    fused = np.array([np.mean(list(prob_by_pid[pid].values())) for pid in common])
    labels = np.array([label_by_pid[pid] for pid in common])
    return fused, labels, np.array(common)


#################################
# Plotting Curves

def plot_curves(labels, predictions, title, save_dir):
    """ROC (left) and PR (right) for a single model, saved to {save_dir}/{title}_curves.png."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, predictions)
    prec, rec, _ = precision_recall_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    ap = average_precision_score(labels, predictions)
    prevalence = np.mean(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax1.plot([0, 1], [0, 1], '--', color='grey')
    ax1.set_xlabel('1 - Specificity'); ax1.set_ylabel('Sensitivity')
    ax1.set_title('ROC'); ax1.legend(loc='lower right')

    ax2.plot(rec, prec, label=f'AP = {ap:.3f}')
    ax2.axhline(prevalence, ls='--', color='grey', label=f'Prevalence = {prevalence:.3f}')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall'); ax2.legend(loc='lower left')

    fig.suptitle(title)
    fig.tight_layout()
    out = Path(save_dir) / f'{title}_curves.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    return out

# Plotting Compaison Curves
def plot_comparison_curves(curves, title, save_dir):
    """Overlay several models on one ROC and one PR axis.
    curves: list of (name, labels, predictions)."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, labels, preds in curves:
        fpr, tpr, _ = roc_curve(labels, preds)
        prec, rec, _ = precision_recall_curve(labels, preds)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(labels, preds):.3f})')
        ax2.plot(rec, prec, label=f'{name} (AP={average_precision_score(labels, preds):.3f})')
    ax1.plot([0, 1], [0, 1], '--', color='grey')
    ax1.set_xlabel('1 - Specificity'); ax1.set_ylabel('Sensitivity'); ax1.set_title('ROC'); ax1.legend(loc='lower right')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('Precision-Recall'); ax2.legend(loc='lower left')
    fig.suptitle(title); fig.tight_layout()
    out = Path(save_dir) / f'{title}_comparison.png'
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    return out


#######################
# Evaluating per-mode


def evaluate_mode(condition, architecture, data_mode, args, device):
    """Evaluate one (architecture, data_mode) cell: per-plane + fused.
    Returns None if no plane models are available, else a results dict."""
    print(f"\n=== {condition} | {ARCH_LABELS[architecture]} | {data_mode} ===")
    plane_results = {}
    val_plane_results = {}   # val predictions which are only collected if we are tuning
    for plane in PLANES:
        model = load_single_model(condition, plane, architecture, data_mode, args.checkpoint_dir, device)
        if model is None:
            continue
        dataset = ValidMRNetDataset(args.data_dir, condition, plane, data_mode)
        if len(dataset) == 0:
            print(f"  [warn] empty test set for {plane}; check data_dir/data_mode")
            continue
        plane_results[plane] = get_plane_predictions(model, dataset, device)

        #  reusing the model that is already loaded to predict on the val split
        if args.tune_threshold:
            val_ds = MRNetDataset(args.data_dir, condition, plane, 'val', data_mode, transform=None)
            val_plane_results[plane] = get_plane_predictions(model, val_ds, device)

    if not plane_results:
        return None

    # choosing the operating threshold (either tuned on val or fixed default)
    threshold = args.threshold
    if args.tune_threshold and val_plane_results:
        vp, vl, _ = fuse_planes(val_plane_results)
        threshold = optimal_threshold(vl, vp)
        print(f"  [threshold] tuned on validation (Youden J) = {threshold:.3f}")

    per_plane = {}
    for plane, (probs, labels, _) in plane_results.items():
        per_plane[plane] = compute_metrics_with_ci(labels, probs, threshold, args.n_bootstraps, args.seed)

    fused_probs, fused_labels, fused_pids = fuse_planes(plane_results)
    fused_metrics = compute_metrics_with_ci(fused_labels, fused_probs, threshold, args.n_bootstraps, args.seed)

    title = f'{condition}_{architecture}_{data_mode}_fused'
    plot_curves(fused_labels, fused_probs, title, args.output_dir)

    return {
        'condition': condition,
        'architecture': architecture,
        'data_mode': data_mode,
        'threshold': threshold,   # to record which threshold was actually used
        'per_plane_metrics': per_plane,
        'fused_metrics': fused_metrics,
        'fused_raw': {'pids': fused_pids, 'probs': fused_probs, 'labels': fused_labels},
        'plane_raw': plane_results,
    }


#########################################################
# Matrix (2x2) and main/interaction effects for fused AUC

def build_2x2_matrix(results):
    """results: {(architecture, data_mode): mode_result}. 
    Returns matrix + effect estimates."""
    def fused_auc(arch, mode):
        r = results.get((arch, mode))
        return r['fused_metrics']['auc'][0] if r else None

    cells = {f'{a}_{m}': fused_auc(a, m)
             for a in ['baseline', 'comparative'] for m in ['uncropped', 'cropped']}

    effects = {}
    bu, bc = cells['baseline_uncropped'], cells['baseline_cropped']
    cu, cc = cells['comparative_uncropped'], cells['comparative_cropped']
    if None not in (bu, bc, cu, cc):
        effects['cropping_main'] = ((bc + cc) - (bu + cu)) / 2  # mean auc gain from cropping, that saveraged over both architectures
        effects['pretraining_main'] = ((cu + cc) - (bu + bc)) / 2 # mean auc gain from pretraining (RadImageNet), averaged over both data modes
        effects['interaction'] = (cc - cu) - (bc - bu) # does cropping help the comparitive model more than the baseline?
    else:
        effects['note'] = 'Full 2x2 effects need all four cells (comparative requires Role 3).'

    return {'cells_fused_auc': cells, 'effects': effects}


#######################################
# Exporting triage for Role 6

def export_predictions_for_triage(mode_result, output_path):
    """Per-exam JSON along wth per-plane and fused confidences for triage."""
    plane_raw = mode_result['plane_raw']
    prob_by_pid, label_by_pid = {}, {}
    for plane, (probs, labels, pids) in plane_raw.items():
        for prob, lab, pid in zip(probs, labels, pids):
            prob_by_pid.setdefault(int(pid), {})[plane] = float(prob)
            label_by_pid[int(pid)] = int(lab)

    fused = dict(zip(mode_result['fused_raw']['pids'].tolist(),
                     mode_result['fused_raw']['probs'].tolist()))

    cases = []
    for pid in sorted(prob_by_pid):
        cases.append({
            'patient_id': pid,
            'true_label': label_by_pid[pid],
            'plane_probs': prob_by_pid[pid],
            'fused_prob': fused.get(pid),
        })
    cases.sort(key=lambda c: (c['fused_prob'] is None, -(c['fused_prob'] or 0)))

    payload = {
        'condition': mode_result['condition'],
        'architecture': mode_result['architecture'],
        'data_mode': mode_result['data_mode'],
        'threshold': mode_result.get('threshold', 0.5),
        'cases': cases,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  [triage] exported {len(cases)} cases -> {output_path}")


###################################################
# serialize results to JSON and print a summary table

def _fmt(metric_ci):
    v, lo, hi = metric_ci
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def save_results(condition, results, matrix, output_dir):
    serialisable = {}
    for (arch, mode), r in results.items():
        serialisable[f'{arch}_{mode}'] = {
            'threshold': r.get('threshold', 0.5),   #persisrt the threshold used 
            'per_plane': {p: {m: list(map(float, ci)) for m, ci in mets.items()}
                          for p, mets in r['per_plane_metrics'].items()},
            'fused': {m: list(map(float, ci)) for m, ci in r['fused_metrics'].items()},
        }
    out = {'condition': condition, 'modes': serialisable, 'matrix_2x2': matrix}
    path = Path(output_dir) / f'results_{condition}.json'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics -> {path}")

    print(f"\n{'='*70}\nFused results: {condition}\n{'='*70}")
    for (arch, mode), r in results.items():
        fm = r['fused_metrics']
        print(f"{ARCH_LABELS[arch]:>22} | {mode:<9} | "
              f"AUC {_fmt(fm['auc'])}  Sens {_fmt(fm['sensitivity'])}  "
              f"Spec {_fmt(fm['specificity'])}  F1 {_fmt(fm['f1'])}")
    print(f"\n2x2 fused AUC: {matrix['cells_fused_auc']}")
    print(f"Effects: {matrix['effects']}")


###################
# Orchaestrating

def evaluate_condition(args, device):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for arch in args.architectures:
        for mode in args.data_modes:
            r = evaluate_mode(args.condition, arch, mode, args, device)
            if r is not None:
                results[(arch, mode)] = r

    if not results:
        print("\nNo evaluable models found. Have any been trained yet?")
        return results   # an empty dict allows the master table wrapper to skip it easier

    # Overlay curves
    #RQ2: pretraining effect (baseline vs comparitive models) for each mode 
    for mode in args.data_modes:
        curves = []
        for arch in args.architectures:
            r = results.get((arch, mode))
            if r:
                curves.append((ARCH_LABELS[arch], r['fused_raw']['labels'], r['fused_raw']['probs']))
        if len(curves) > 1:
            plot_comparison_curves(curves, f'{args.condition}_{mode}_arch_overlay', args.output_dir)

    #Paired significance tests for same paients for both research questions
    sig = {}
    for arch in args.architectures:
        ru, rc = results.get((arch, 'uncropped')), results.get((arch, 'cropped'))
        if ru and rc:
            lab, a, b = _aligned(ru['fused_raw'], rc['fused_raw'])
            sig[f'cropping_effect_{arch}'] = _sig(lab, a, b, args)
    for mode in args.data_modes:
        rb, rc = results.get(('baseline', mode)), results.get(('comparative', mode))
        if rb and rc:
            lab, a, b = _aligned(rb['fused_raw'], rc['fused_raw'])
            sig[f'pretraining_effect_{mode}'] = _sig(lab, a, b, args)
    if sig:
        with open(Path(args.output_dir) / f'significance_{args.condition}.json', 'w') as f:
            json.dump(sig, f, indent=2)
        print(f"\nPaired AUC-difference tests: {sig}")

    matrix = build_2x2_matrix(results)
    save_results(args.condition, results, matrix, args.output_dir)

    # Triage export: best available cell by fused AUC
    best_key = max(results, key=lambda k: results[k]['fused_metrics']['auc'][0])
    export_predictions_for_triage(
        results[best_key],
        Path(args.output_dir) / f'{args.condition}_triage.json')

    return results   #so that evaluate_all_conditions can aggregate them

def _aligned(raw_a, raw_b):
    """Align two fused-prediction sets on common patient ids."""
    a = dict(zip(raw_a['pids'].tolist(), raw_a['probs'].tolist()))
    b = dict(zip(raw_b['pids'].tolist(), raw_b['probs'].tolist()))
    lab = dict(zip(raw_a['pids'].tolist(), raw_a['labels'].tolist()))
    common = sorted(set(a) & set(b))
    return (np.array([lab[p] for p in common]),
            np.array([a[p] for p in common]),
            np.array([b[p] for p in common]))


def _sig(labels, probs_a, probs_b, args):
    diff, lo, hi, p = paired_bootstrap_auc_diff(labels, probs_a, probs_b, args.n_bootstraps, args.seed)
    return {'auc_diff_b_minus_a': diff, 'ci_low': lo, 'ci_high': hi,
            'p_two_sided': p, 'significant_at_0.05': bool(lo > 0 or hi < 0)} #significant if 95% CI for auc diff excludes 0


# wrapper for master table generation across all conditions and combined csv/json
def evaluate_all_conditions(args, device):
    master = []
    for cond in ['acl', 'meniscus', 'abnormal']:
        sub = argparse.Namespace(**vars(args))
        sub.condition = cond
        results = evaluate_condition(sub, device)
        if not results:
            continue
        for (arch, mode), r in results.items():
            fm = r['fused_metrics']
            master.append({
                'condition': cond, 'architecture': arch, 'data_mode': mode,
                'threshold': round(r.get('threshold', 0.5), 4),
                'auc': round(fm['auc'][0], 4),
                'auc_ci_low': round(fm['auc'][1], 4), 'auc_ci_high': round(fm['auc'][2], 4),
                'sensitivity': round(fm['sensitivity'][0], 4),
                'specificity': round(fm['specificity'][0], 4),
                'f1': round(fm['f1'][0], 4),
            })

    if master:
        csv_path = Path(args.output_dir) / 'master_table.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(master[0].keys()))
            writer.writeheader()
            for row in master:
                writer.writerow(row)
        with open(Path(args.output_dir) / 'master_table.json', 'w') as f:
            json.dump(master, f, indent=2)
        print(f"\nMaster table ({len(master)} rows) -> {csv_path}")
    else:
        print("\nNo results to aggregate into a master table.")
    return master


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating condition='{args.condition}' on {device}")
    # dispatch 'all' to maste tabel or else eval one condition
    if args.condition == 'all':
        evaluate_all_conditions(args, device)
    else:
        evaluate_condition(args, device)
    print(f"\nDone. Results in {args.output_dir}")


if __name__ == '__main__':
    main()