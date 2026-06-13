# MRNet Project: Roles and Responsibilities

## Project Overview

This project builds a deep learning pipeline to classify knee MRI exams for ACL tears, meniscal tears, and general abnormalities. See [Initial Research Plan](initial_research_plan.md) for detailed methodology.

### Novel Contributions

1. **R-CNN Semantic Cropping:** Faster R-CNN preprocessing to focus on joint space (trained on manual annotations)
2. **Medical Imaging Pretraining Comparison:** RadImageNet vs ImageNet weights
3. **Clinical Triage Interface:** Confidence-based case prioritization with slice-level explainability

### Experimental Design

**2×2 Comparison Matrix:**
- **Architectures:** ResNet18-ImageNet (baseline) vs ResNet50-RadImageNet (comparative)
- **Data Modes:** Uncropped (original) vs Cropped (R-CNN preprocessed)
- **Total Models:** 36 (4 modes × 3 conditions × 3 planes)

### Key Decisions

- **Training:** Single-phase end-to-end fine-tuning with small learning rate (~1e-4)
- **Augmentation:** Contrast/brightness jitter + small rotations (±10°), no flips
- **Data Split:** 80/20 split of `train/` folder, held-out `valid/` folder as test set
- **Checkpointing:** Best model only (based on validation AUC)
- **Hyperparameters:** Tune once per architecture, apply to both data modes
- **Preprocessing:** One-time R-CNN cropping saved to `data/mrnet_cropped/`
- **Compute:** SLURM HPC, parallel training across all models

---

## Role 1: Data Preprocessing & Transformations

**File:** `src/modules/data_preprocessing_transformation.py`

### Responsibilities

1. **Dataset Implementation**
   - Parameterized dataset: `Dataset(condition, plane, split, data_mode='uncropped'/'cropped')`
   - Returns variable-length tensors: `(num_slices, 3, H, W)` (grayscale stacked to 3 channels)
   - Batch size = 1 (no batching due to variable slice counts)

2. **Data Augmentation**
   - Contrast/brightness jittering (simulates different MRI scanners)
   - Random rotations (±10° to preserve anatomy)
   - **No horizontal/vertical flips** (preserves anatomical sidedness)
   - Applied on-the-fly during training

3. **Data Modes**
   - `uncropped`: Load from `data/mrnet/` (original data)
   - `cropped`: Load from `data/mrnet_cropped/` (R-CNN preprocessed)

4. **Class Imbalance**
   - Compute `pos_weight` for BCEWithLogitsLoss: `num_negative / num_positive`
   - Expose as `dataset.pos_weight` attribute

5. **R-CNN Preprocessing Pipeline**
   - Run `crop_mrnet_dataset.py` once before training
   - Saves cropped volumes to `data/mrnet_cropped/` with same directory structure

### Interface

```python
# Standard dataset usage
dataset = Dataset(condition='ACL', plane='axial', split='train', data_mode='cropped')
exam, label = dataset[idx]  # Returns (num_slices, 3, H, W) and label
pos_weight = dataset.pos_weight  # For loss function
```

---

## Role 2: Baseline Models

**File:** `src/modules/baseline_models.py`

### Responsibilities

1. **Model Architecture**
   - ResNet18 (ImageNet pretrained) feature extractor → Max pooling → Binary classifier
   - Architecture: `(num_slices, 3, H, W)` → ResNet18 → `(num_slices, 512)` → Max pool → `(512)` → FC → `(1)` logit

2. **Base Class Implementation**
   - `MRNetBaseModel` base class (reused by Role 3 for comparative models)
   - Encapsulates: max pooling, custom FC head, 1→3 channel handling

3. **Slice Tracking for Explainability**
   - `forward_with_slice_tracking(x)` returns `(logits, slice_indices)`
   - Tracks which slices contributed to max pooling for each feature dimension
   - Used by Role 6 for Grad-CAM visualization

4. **Model Factory**
   - `create_baseline_model()` returns configured ResNet18 instance

### Interface

```python
model = create_baseline_model()
logits = model(exam_tensor)  # Standard forward pass

# For explainability (Role 6 uses this)
logits, slice_indices = model.forward_with_slice_tracking(exam_tensor)
# slice_indices: (512,) tensor showing which slice was max for each feature
```

---

## Role 3: Comparative Models

**File:** `src/modules/comparative_models.py`

### Responsibilities

1. **Alternative Architecture**
   - ResNet50 with RadImageNet pretraining (medical imaging weights)
   - Deeper network, domain-specific pretrained features

2. **Code Reuse**
   - Inherits from `MRNetBaseModel` base class (Role 2)
   - Same interface: max pooling, slice tracking, FC head
   - Only swaps the backbone network

3. **Model Factory**
   - `create_comparative_model()` returns configured ResNet50-RadImageNet instance
   - Same API as baseline for architecture-agnostic training code

### Interface

```python
model = create_comparative_model()
logits = model(exam_tensor)  # Same interface as baseline

# Slice tracking inherited from MRNetBaseModel
logits, slice_indices = model.forward_with_slice_tracking(exam_tensor)
```

---

## Role 4: Training Pipeline & Hyperparameter Tuning

**File:** `src/train.py`

### Responsibilities

1. **Training Orchestration**
   - Modular `train_single.py` script: trains ONE model
   - Arguments: `--condition`, `--plane`, `--architecture`, `--data_mode`
   - SLURM batch submission for parallel training across all 36 models

2. **Training Strategy**
   - Single-phase end-to-end fine-tuning with small learning rate (~1e-4)
   - Train entire model (backbone + head) from start
   - No freeze/unfreeze phases

3. **Data Split Management**
   - 80/20 split of `train/` folder → train/validation
   - Validation for early stopping and model selection
   - Held-out `valid/` folder as final test set (for Role 5)
   - Consistent random seed for reproducibility

4. **Hyperparameter Tuning**
   - Tune once per architecture (2 tuning runs total)
   - Apply tuned hyperparameters to all models using that architecture
   - Tune: learning rate, epochs, optimizer, weight decay

5. **Loss Function**
   - `BCEWithLogitsLoss(pos_weight=dataset.pos_weight)`
   - Different `pos_weight` per condition (from Role 1)

6. **Checkpointing**
   - Save best model only (based on validation AUC)
   - Structure: `checkpoints/{condition}_{plane}_{architecture}_{data_mode}/`
   - Include metadata: hyperparameters, training time, final metrics

7. **Experiment Tracking**
   - File-based checkpoints with config YAML
   - TensorBoard for training curves

### Interface

```bash
# Train a single model
python train_single.py --condition ACL --plane axial \
                       --architecture baseline --data_mode cropped

# Submit all 36 models to SLURM
bash submit_all.sh  # Trains all combinations in parallel
```

---

## Role 5: Evaluation & Metrics

**File:** `src/evaluation.py`

### Responsibilities

1. **Multi-Plane Prediction Fusion**
   - Load 3 plane models per condition (axial, coronal, sagittal)
   - Fuse predictions: `final_prob = (prob_axial + prob_coronal + prob_sagittal) / 3`

2. **Metrics Suite**
   - AUC (primary metric), Sensitivity, Specificity, F1-Score
   - All metrics with 95% confidence intervals (bootstrap)
   - Threshold: 0.5 for binary metrics

3. **Multi-Level Reporting**
   - Per-plane results (individual plane performance)
   - Fused results per condition (3-plane combined)
   - **2×2 Comparison Matrix:**
     - Main effect: Cropping (uncropped vs cropped)
     - Main effect: Pretraining (ImageNet vs RadImageNet)
     - Interaction effects

4. **Visualization**
   - ROC curves per condition
   - Precision-Recall curves (important for imbalanced classes)
   - Overlay baseline vs comparative models

5. **Prediction Export for Triage**
   - Export per-exam predictions with per-plane AND fused confidences
   - JSON format for Role 6 triage interface

### Interface

```python
# Evaluate all 4 training modes for one condition
results = evaluate_all_modes(condition='ACL', test_loader)
# Returns unified comparison table with confidence intervals

# Export for triage tool
export_predictions_for_triage(predictions, 'predictions/acl_triage.json')
```

---

## Role 6: Explainability & Clinical Presentation

**File:** `src/explainability.py`

### Responsibilities

1. **Grad-CAM Implementation**
   - Generate heatmaps on most informative slices (identified via slice tracking)
   - Overlay on MRI slices to show model attention

2. **Clinical Triage Interface**
   - **Confidence-Based Ranking:** Sort cases by fused confidence (high to low)
   - **Multi-Plane Visualization:** Display all 3 planes with Grad-CAM overlays
   - **Per-Plane Confidence:** Show individual plane scores and fused score
   - **Most Informative Slice:** Display the slice that contributed most to prediction

3. **Scope**
   - Primary: ACL tear detection (all 3 planes)
   - Stretch: Meniscus and Abnormal conditions if time permits
   - Use best performing model from evaluation

4. **Implementation Format**
   - Jupyter notebook with interactive widgets OR simple HTML interface
   - Display top-N cases (e.g., N=20-50 highest confidence predictions)

### Interface

```python
# Identify most informative slice for an exam
slice_idx = identify_most_informative_slice(model, exam)

# Generate Grad-CAM for that slice
heatmap = generate_gradcam(model, exam, slice_idx)

# Create triage interface
create_triage_interface(predictions_json, models, dataset, top_n=50)
# Generates interactive visualization sorted by confidence
```

---

## Summary

This streamlined roles document focuses on essential responsibilities and interfaces. For detailed architectural decisions and terminology, see [Initial Research Plan](initial_research_plan.md) and [CONTEXT.md](../CONTEXT.md).