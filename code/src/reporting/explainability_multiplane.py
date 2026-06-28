"""
Role 6: Multi-Plane Explainability & Clinical Triage Interface

Generates comprehensive HTML triage reports with:
- Multi-plane Grad-CAM visualizations (axial, coronal, sagittal)
- Fused prediction confidence breakdown
- Plane agreement/disagreement analysis
- Priority-ranked patient cases
- Statistical performance context

Usage:
    python3 explainability_multiplane.py \
        --triage_json results/acl_triage.json \
        --results_json results/results_acl.json \
        --significance_json results/significance_acl.json \
        --condition acl \
        --architecture baseline \
        --data_mode cropped \
        --checkpoint_dir checkpoints/ \
        --data_dir code/src/data/ \
        --output_html job_outputs/acl_multiplane_triage.html
"""

import argparse
import json
import base64
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import mode
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.data_preprocessing_transformation import ValidMRNetDataset
from modules.baseline_models import create_baseline_model


class SingleSliceWrapper(nn.Module):
    """Wrapper to make MRNetBaseModel compatible with GradCAM."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.backbone = model.backbone
        self.fc = model.fc
    
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.model(x)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate multi-plane Grad-CAM triage report')
    parser.add_argument('--triage_json', type=str, required=True, help='Path to triage JSON from evaluation')
    parser.add_argument('--results_json', type=str, required=True, help='Path to results JSON with metrics')
    parser.add_argument('--significance_json', type=str, required=True, help='Path to significance test JSON')
    parser.add_argument('--condition', type=str, required=True, choices=['acl', 'meniscus', 'abnormal'])
    parser.add_argument('--architecture', type=str, default='baseline', choices=['baseline', 'comparative'])
    parser.add_argument('--data_mode', type=str, required=True, choices=['uncropped', 'cropped'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--data_dir', type=str, default='src/data/')
    parser.add_argument('--output_html', type=str, default='job_outputs/multiplane_triage.html')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top cases to visualize')
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_triage_data(triage_json_path):
    """Load triage JSON with patient rankings."""
    with open(triage_json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['cases'])} cases from triage JSON")
    return data


def load_results_metadata(results_json_path, significance_json_path):
    """Load performance metrics and statistical tests."""
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    with open(significance_json_path, 'r') as f:
        significance = json.load(f)
    
    return results, significance


def load_model_for_plane(condition, plane, architecture, data_mode, checkpoint_dir, device):
    """Load a single plane model."""
    model_name = f"{condition}_{plane}_{architecture}_{data_mode}"
    model_dir = Path(checkpoint_dir) / model_name
    
    # Find checkpoint
    for ckpt_name in ['best_phase2.pth', 'best_single_phase.pth', 'best_phase1.pth']:
        ckpt_path = model_dir / ckpt_name
        if ckpt_path.exists():
            break
    else:
        print(f"  [ERROR] No checkpoint found for {plane}")
        return None
    
    # Load model
    model = create_baseline_model()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    
    print(f"  [OK] Loaded {plane} model from {ckpt_path.name}")
    return model


def load_all_plane_models(condition, architecture, data_mode, checkpoint_dir, device):
    """Load all 3 plane models."""
    print(f"\nLoading models for {condition}-{architecture}-{data_mode}:")
    models = {}
    for plane in ['axial', 'coronal', 'sagittal']:
        model = load_model_for_plane(condition, plane, architecture, data_mode, checkpoint_dir, device)
        if model is not None:
            models[plane] = model
    return models


def load_datasets(condition, data_mode, data_dir):
    """Load validation datasets for all 3 planes."""
    print(f"\nLoading validation datasets:")
    datasets = {}
    for plane in ['axial', 'coronal', 'sagittal']:
        dataset = ValidMRNetDataset(data_dir, condition, plane, data_mode)
        datasets[plane] = dataset
        print(f"  [OK] {plane}: {len(dataset)} cases")
    return datasets


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_plane_agreement(plane_probs):
    """Calculate agreement score and categorize confidence level."""
    probs = [plane_probs.get('axial', 0.5), plane_probs.get('coronal', 0.5), plane_probs.get('sagittal', 0.5)]
    disagreement = max(probs) - min(probs)
    
    if disagreement < 0.15:
        level = 'high'
        emoji = '[HIGH]'
        description = f'High Agreement - All planes converge (variance: {disagreement:.1%})'
        css_class = 'high-agreement'
    elif disagreement < 0.35:
        level = 'moderate'
        emoji = '[MOD]'
        description = f'Moderate Agreement - Some variance across planes ({disagreement:.1%})'
        css_class = 'moderate-agreement'
    else:
        level = 'low'
        emoji = '[LOW]'
        description = f'Low Agreement - Planes disagree significantly ({disagreement:.1%})'
        css_class = 'low-agreement'
    
    return {
        'disagreement': disagreement,
        'level': level,
        'emoji': emoji,
        'description': description,
        'css_class': css_class
    }


def find_most_informative_slice(model, exam, device):
    """Run forward_with_slice_tracking to find which slice contributed most."""
    with torch.no_grad():
        exam_input = exam.unsqueeze(0).to(device)
        logits, slice_indices, pooled = model.forward_with_slice_tracking(exam_input)
        
        # Filter out features where max pooled activation is exactly 0 
        # (Otherwise torch.max defaults to slice 0 for all zero-channels, skewing the mode)
        idxs = slice_indices.cpu().numpy().flatten()
        vals = pooled.cpu().numpy().flatten()
        non_zero_idxs = idxs[vals > 0]
        
        if len(non_zero_idxs) > 0:
            most_informative_slice = int(mode(non_zero_idxs, keepdims=False)[0])
            contribution_count = int((non_zero_idxs == most_informative_slice).sum())
            contribution_pct = (contribution_count / len(non_zero_idxs)) * 100
        else:
            most_informative_slice = int(mode(idxs, keepdims=False)[0])
            contribution_count = int((idxs == most_informative_slice).sum())
            contribution_pct = (contribution_count / len(idxs)) * 100
    
    return {
        'slice_idx': most_informative_slice,
        'contribution_count': contribution_count,
        'contribution_pct': contribution_pct,
        'total_slices': exam.shape[0]
    }


# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

def get_gradcam_heatmap(model, exam, target_slice_idx, device):
    """Generate Grad-CAM heatmap for a specific slice."""
    wrapped_model = SingleSliceWrapper(model)
    target_layers = [wrapped_model.backbone[-2]]
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    
    slice_input = exam[:, target_slice_idx, :, :, :].to(device)
    grayscale_cam = cam(input_tensor=slice_input)
    
    return grayscale_cam[0]


def generate_gradcam_for_plane(model, exam, slice_idx, device):
    """Generate Grad-CAM visualizations for one plane."""
    # Get original slice
    original = exam[slice_idx, 0, :, :].cpu().numpy()
    
    # Generate heatmap
    heatmap = get_gradcam_heatmap(model, exam.unsqueeze(0), slice_idx, device)
    
    # Create overlay
    overlay = show_cam_on_image(
        np.stack([original] * 3, axis=2),
        heatmap,
        use_rgb=True
    )
    
    return {
        'original': original,
        'heatmap': heatmap,
        'overlay': overlay
    }


def image_to_base64(img_array):
    """Convert numpy array to base64 encoded PNG."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_array)
    ax.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return img_b64


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_header_section(results, significance, condition, architecture, data_mode):
    """Generate header with model info and performance metrics."""
    # Extract metrics from results JSON
    mode_key = f"{architecture}_{data_mode}"
    fused_metrics = results['modes'][mode_key]['fused']
    
    auc = fused_metrics['auc']
    sens = fused_metrics['sensitivity']
    spec = fused_metrics['specificity']
    f1 = fused_metrics['f1']
    
    # Architecture label
    arch_label = 'AlexNet-ImageNet' if architecture == 'baseline' else 'ResNet50-RadImageNet'
    
    # Significance test
    sig_key = f'cropping_effect_{architecture}'
    if sig_key in significance:
        sig_data = significance[sig_key]
        auc_diff = sig_data['auc_diff_b_minus_a'] * 100
        p_val = sig_data['p_two_sided']
        is_sig = sig_data['significant_at_0.05']
        sig_text = f"+{auc_diff:.1f}% AUC improvement (p={p_val:.3f}, {'significant' if is_sig else 'not significant'} at α=0.05)"
    else:
        sig_text = "Significance test not available"
    
    html = f"""
    <div class="header">
        <h1>{condition.upper()} Detection - Multi-Plane Triage Report</h1>
        <p style="font-size: 16px; color: #7f8c8d;">
            <strong>Model:</strong> {arch_label} | 
            <strong>Data:</strong> {data_mode.capitalize()} | 
            <strong>Threshold:</strong> 0.50
        </p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" style="color: #3498db;">{auc[0]:.3f}</div>
                <div class="metric-label">Fused AUC</div>
                <div class="metric-ci">[{auc[1]:.3f} - {auc[2]:.3f}]</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #2ecc71;">{sens[0]*100:.1f}%</div>
                <div class="metric-label">Sensitivity</div>
                <div class="metric-ci">[{sens[1]*100:.1f}% - {sens[2]*100:.1f}%]</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #e74c3c;">{spec[0]*100:.1f}%</div>
                <div class="metric-label">Specificity</div>
                <div class="metric-ci">[{spec[1]*100:.1f}% - {spec[2]*100:.1f}%]</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #f39c12;">{f1[0]:.3f}</div>
                <div class="metric-label">F1 Score</div>
                <div class="metric-ci">[{f1[1]:.3f} - {f1[2]:.3f}]</div>
            </div>
        </div>
        
        <div class="stat-context">
            <strong>Cropping Effect:</strong> {sig_text}
        </div>
    </div>
    """
    return html


def generate_confidence_breakdown(plane_probs, fused_prob):
    """Generate horizontal bar chart showing per-plane contributions."""
    html = """
    <div class="confidence-bars">
        <h4>Prediction Breakdown by Plane</h4>
    """
    
    planes = [
        ('Axial', plane_probs.get('axial', 0.5), '#3498db', '#2980b9'),
        ('Coronal', plane_probs.get('coronal', 0.5), '#f39c12', '#e67e22'),
        ('Sagittal', plane_probs.get('sagittal', 0.5), '#9b59b6', '#8e44ad'),
    ]
    
    for plane_name, prob, color1, color2 in planes:
        html += f"""
        <div class="confidence-bar">
            <div class="bar-label">{plane_name}</div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: {prob*100:.1f}%; background: linear-gradient(90deg, {color1}, {color2});">
                    {prob*100:.1f}%
                </div>
            </div>
        </div>
        """
    
    html += f"""
        <div class="confidence-bar fused-bar">
            <div class="bar-label"><strong>Fused Prediction</strong></div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: {fused_prob*100:.1f}%; background: linear-gradient(90deg, #2ecc71, #27ae60);">
                    <strong>{fused_prob*100:.1f}%</strong>
                </div>
            </div>
        </div>
    </div>
    """
    return html


def generate_case_section(case_data, models, datasets, device, case_num, total_cases, condition):
    """Generate complete visualization for one patient case."""
    patient_id = case_data['patient_id']
    true_label = case_data['true_label']
    plane_probs = case_data['plane_probs']
    fused_prob = case_data['fused_prob']
    
    # Calculate agreement
    agreement = calculate_plane_agreement(plane_probs)
    
    # Label text
    condition_upper = condition.upper()
    if true_label == 1:
        label_html = f'<span class="label-badge positive"> True Positive ({condition_upper} Present)</span>'
    else:
        label_html = f'<span class="label-badge negative"> True Negative (No {condition_upper})</span>'
    
    html = f"""
    <div class="case">
        <div class="case-header">
            <div>
                <h3>Patient {patient_id}</h3>
                <span class="case-number">Case #{case_num} of {total_cases}</span>
            </div>
            <div class="fused-confidence-display">
                <div class="fused-value">{fused_prob*100:.1f}%</div>
                <div class="fused-label">Fused Confidence</div>
            </div>
        </div>
        
        <div class="case-badges">
            {label_html}
            <span class="agreement-badge {agreement['css_class']}">
                {agreement['emoji']} {agreement['level'].capitalize()} Agreement ({agreement['disagreement']:.1%} variance)
            </span>
        </div>
    """
    
    # Generate multi-plane grid
    html += '<div class="plane-grid">'
    
    for plane in ['axial', 'coronal', 'sagittal']:
        prob = plane_probs.get(plane, 0.5)
        model = models[plane]
        dataset = datasets[plane]
        
        # Find patient in dataset
        patient_idx = None
        for idx in range(len(dataset)):
            if int(Path(dataset.file_paths[idx]).stem) == patient_id:
                patient_idx = idx
                break
        
        if patient_idx is None:
            html += f'<div class="plane"><p>Patient {patient_id} not found in {plane} dataset</p></div>'
            continue
        
        # Load exam
        exam, _ = dataset[patient_idx]
        
        # Find most informative slice
        slice_info = find_most_informative_slice(model, exam, device)
        
        # Generate Grad-CAM
        visuals = generate_gradcam_for_plane(model, exam, slice_info['slice_idx'], device)
        img_b64 = image_to_base64(visuals['overlay'])
        
        html += f"""
        <div class="plane">
            <h4 class="plane-title">{plane.capitalize()} Plane</h4>
            <div class="plane-confidence">{prob*100:.1f}%</div>
            <img src="data:image/png;base64,{img_b64}" class="gradcam-img" />
            <div class="slice-info">
                <div><strong>Most Informative Slice:</strong> #{slice_info['slice_idx']} / {slice_info['total_slices']}</div>
                <div><strong>Contribution:</strong> {slice_info['contribution_count']}/256 features ({slice_info['contribution_pct']:.1f}%)</div>
            </div>
        </div>
        """
    
    html += '</div>'  # Close plane-grid
    
    # Add confidence breakdown
    html += generate_confidence_breakdown(plane_probs, fused_prob)
    
    # Add agreement analysis box
    html += f"""
        <div class="agreement-box {agreement['css_class']}">
            <strong>{agreement['emoji']} {agreement['level'].upper()} AGREEMENT ACROSS PLANES</strong><br/>
            <span>{agreement['description']}</span>
        </div>
    </div>
    """
    
    return html


def generate_css():
    """Generate CSS styling for the HTML report."""
    return """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .metric-ci {
            font-size: 12px;
            color: #95a5a6;
        }
        .stat-context {
            margin-top: 20px;
            padding: 15px;
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }
        h2 {
            color: #2c3e50;
            margin: 40px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        .case {
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .case-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }
        .case-header h3 {
            margin: 0;
            color: #2c3e50;
            font-size: 24px;
        }
        .case-number {
            color: #7f8c8d;
            font-size: 14px;
        }
        .fused-confidence-display {
            text-align: right;
        }
        .fused-value {
            font-size: 42px;
            font-weight: bold;
            color: #e74c3c;
        }
        .fused-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        .case-badges {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        .label-badge {
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        .label-badge.positive {
            background: #d4edda;
            color: #155724;
        }
        .label-badge.negative {
            background: #d1ecf1;
            color: #0c5460;
        }
        .agreement-badge {
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        .high-agreement {
            background: #d4edda;
            color: #155724;
        }
        .moderate-agreement {
            background: #fff3cd;
            color: #856404;
        }
        .low-agreement {
            background: #f8d7da;
            color: #721c24;
        }
        .plane-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }
        .plane {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .plane-title {
            color: #3498db;
            margin: 0 0 10px 0;
        }
        .plane-confidence {
            font-size: 28px;
            font-weight: bold;
            color: #e74c3c;
            margin-bottom: 15px;
        }
        .gradcam-img {
            width: 100%;
            border-radius: 4px;
            margin-bottom: 15px;
        }
        .slice-info {
            background: white;
            padding: 12px;
            border-radius: 4px;
            text-align: left;
            font-size: 13px;
        }
        .slice-info div {
            margin: 5px 0;
        }
        .confidence-bars {
            margin: 25px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .confidence-bars h4 {
            color: #2c3e50;
            margin: 0 0 15px 0;
        }
        .confidence-bar {
            margin: 12px 0;
        }
        .fused-bar {
            margin-top: 20px;
            padding-top: 15px;
            border-top: 2px solid #e0e0e0;
        }
        .bar-label {
            font-size: 13px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .bar-bg {
            background: #e0e0e0;
            height: 32px;
            border-radius: 4px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            display: flex;
            align-items: center;
            padding-left: 12px;
            color: white;
            font-weight: bold;
            font-size: 14px;
            transition: width 0.3s ease;
        }
        .agreement-box {
            margin-top: 20px;
            padding: 15px;
            border-left: 4px solid;
            border-radius: 4px;
        }
        .agreement-box.high-agreement {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        .agreement-box.moderate-agreement {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        .agreement-box.low-agreement {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
    </style>
    """


def generate_full_html(triage_data, results, significance, models, datasets, device, condition, architecture, data_mode, top_n=20):
    """Main function that orchestrates full HTML generation."""
    html = ['<!DOCTYPE html>', '<html>', '<head>']
    html.append(f'<title>{condition.upper()} Detection - Multi-Plane Triage Report</title>')
    html.append(generate_css())
    html.append('</head>')
    html.append('<body>')
    
    # Header section
    html.append(generate_header_section(results, significance, condition, architecture, data_mode))
    
    # Get cases
    all_cases = triage_data['cases']
    total_cases = len(all_cases)
    
    # High-priority cases (top N)
    html.append(f'<h2>High-Priority Cases (Top {min(top_n, total_cases)} by Fused Confidence)</h2>')
    for idx, case in enumerate(all_cases[:top_n], 1):
        print(f"  Processing case {idx}/{top_n}: Patient {case['patient_id']}...")
        html.append(generate_case_section(case, models, datasets, device, idx, total_cases, condition))
    
    # Uncertain cases (around threshold)
    uncertain = [c for c in all_cases if 0.4 <= c['fused_prob'] <= 0.6]
    if uncertain:
        html.append(f'<h2>Uncertain Cases (Confidence 40-60%)</h2>')
        for idx, case in enumerate(uncertain[:10], 1):
            case_num = all_cases.index(case) + 1
            print(f"  Processing uncertain case {idx}/10: Patient {case['patient_id']}...")
            html.append(generate_case_section(case, models, datasets, device, case_num, total_cases, condition))
    
    html.append('</head>')
    html.append('<body>')
    
    return '\n'.join(html)


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print(f"Multi-Plane Explainability Report Generator")
    print("="*70)
    print(f"Condition: {args.condition}")
    print(f"Architecture: {args.architecture}")
    print(f"Data mode: {args.data_mode}")
    print(f"Device: {device}")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading triage and results data...")
    triage_data = load_triage_data(args.triage_json)
    results, significance = load_results_metadata(args.results_json, args.significance_json)
    
    # Load models
    print("\n[2/5] Loading plane models...")
    models = load_all_plane_models(args.condition, args.architecture, args.data_mode, args.checkpoint_dir, device)
    
    if len(models) < 3:
        print(f"\n[ERROR] Only {len(models)}/3 plane models loaded. Cannot proceed.")
        return
    
    # Load datasets
    print("\n[3/5] Loading validation datasets...")
    datasets = load_datasets(args.condition, args.data_mode, args.data_dir)
    
    # Generate HTML
    print(f"\n[4/5] Generating Grad-CAM visualizations and HTML report...")
    print(f"  Processing top {args.top_n} cases...")
    
    html_content = generate_full_html(
        triage_data, results, significance, models, datasets, device,
        args.condition, args.architecture, args.data_mode, args.top_n
    )
    
    # Save HTML
    print(f"\n[5/5] Saving HTML report...")
    with open(args.output_html, 'w') as f:
        f.write(html_content)
    
    print("="*70)
    print(f"SUCCESS: Report saved to: {args.output_html}")
    print(f"  Open in a browser to view the multi-plane triage interface")
    print("="*70)


if __name__ == '__main__':
    main()
