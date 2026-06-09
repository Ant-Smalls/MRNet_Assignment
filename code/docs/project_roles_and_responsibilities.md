# MRNet Project: Roles and Responsibilities

## Project Context and Overview

This document outlines the division of work for the MRNet deep learning classification project across 6 team members. The project builds on the architectural decisions documented in the [Initial Research Plan](initial_research_plan.md), which established:

- **9 independent binary models** (3 conditions × 3 planes: ACL, Meniscus, Abnormal across Axial, Coronal, Sagittal)
- **Max pooling** for slice aggregation (baseline)
- **Transfer learning** from ImageNet pre-trained ResNet18
- **Weighted loss functions** to handle class imbalance
- **Medical imaging-specific augmentations** (no horizontal flip, contrast/brightness jitter)

### Project File Structure

```
code/
├── src/
│   ├── modules/
│   │   ├── data_preprocessing_transformation.py  (Role 1)
│   │   ├── baseline_models.py                    (Role 2)
│   │   └── comparative_models.py                 (Role 3)
│   ├── train.py                                   (Role 4)
│   ├── evaluation.py                              (Role 5)
│   └── explainability.py                          (Role 6)
└── docs/
    ├── initial_research_plan.md
    ├── project_roles_and_responsibilities.md
    └── CONTEXT.md
```

### Key Cross-Cutting Decisions

- **Compute Environment:** SLURM HPC with batch job submission, 1 GPU per job
- **Data Split Strategy:** 80/20 split of `train/` folder (train/validation), held-out `valid/` folder as test set
- **Experiment Tracking:** Hybrid approach - file-based checkpoints + TensorBoard for training curves
- **Model Count:** 9 models per architecture (baseline + comparative variants)
- **Shared Code:** `MRNetBaseModel` base class implemented by Role 2, reused by Role 3

---

## Role 1: Data Preprocessing & Transformations

**Owner:** [Team Member Name]  
**File:** `src/modules/data_preprocessing_transformation.py`

### Responsibilities

1. **Dataset Design & Loading**
  - Implement parameterized dataset class: `Dataset(condition, plane, split)`
  - Support all 3 conditions (ACL, Meniscus, Abnormal) and 3 planes (Axial, Coronal, Sagittal)
  - Handle variable-length 3D MRI volumes (slices vary per exam)
2. **Data Handling**
  - Return variable-length 3D tensors: `(num_slices, height, width)`
  - Batch size: 1 (no batching to avoid padding complexity)
  - Convert 1-channel grayscale to 3-channel RGB by stacking (for ImageNet weights)
3. **Data Augmentation**
  - Implement on-the-fly augmentation (applied during training, not pre-computed)
  - **Remove** horizontal flips (preserves medical left/right anatomy)
  - **Add** contrast and brightness jittering (simulates different MRI scanners)
  - Apply augmentations independently to each plane
4. **Class Imbalance Handling**
  - Compute `pos_weight` for BCEWithLogitsLoss during dataset initialization
  - Calculate from training data only: `pos_weight = num_negative / num_positive`
  - Expose as dataset attribute: `dataset.pos_weight`

### Interface to Other Roles

**Outputs:**

- Dataset instances for each condition-plane-split combination
- `pos_weight` attribute accessible to Role 4

**Key Methods:**

```python
dataset = Dataset(condition='ACL', plane='axial', split='train')
exam, label = dataset[idx]  # Returns (num_slices, H, W) tensor and label
pos_weight = dataset.pos_weight  # For loss function
```

### Key Decisions Made

- **Dataset structure:** Single parameterized class (not separate classes per condition)
- **Slice handling:** Variable-length tensors (no padding/truncation)
- **Batching:** Batch size = 1
- **Augmentation timing:** On-the-fly during training
- **Weight computation:** Calculated in dataset `__init__`, per condition

---

## Role 2: Baseline Models

**Owner:** [Team Member Name]  
**File:** `src/modules/baseline_models.py`

### Responsibilities

1. **Model Architecture**
  - Implement ImageNet pre-trained ResNet18 as feature extractor
  - Architecture flow:
    - Input: `(num_slices, 1, H, W)` → Stack to `(num_slices, 3, H, W)`
    - ResNet18 feature extractor → `(num_slices, 512)` features
    - Max pooling across slices → `(512,)` aggregated features
    - Custom FC layer → `(1,)` logit for binary classification
2. **Shared Base Class Implementation**
  - Implement `MRNetBaseModel` base class
  - Encapsulates: max pooling logic, custom FC head, 1→3 channel conversion
  - Provides freeze/unfreeze utilities for transfer learning
  - This base class will be reused by Role 3
3. **Transfer Learning Interface**
  - Provide methods:
    - `freeze_backbone()` - freeze ResNet18 pretrained layers
    - `unfreeze_backbone(num_layers=None)` - unfreeze all or last N layers
    - `get_trainable_params()` - return parameters for optimizer
4. **Model Factory Function**
  - Implement: `create_baseline_model(condition=None, plane=None)`
  - Returns model instance with baseline ResNet18 architecture
  - All 9 models share identical architecture (just trained on different data)

### Interface to Other Roles

**Outputs:**

- `MRNetBaseModel` base class (used by Role 3)
- Factory function for creating model instances
- Freeze/unfreeze utilities (used by Role 4)

**Key Methods:**

```python
model = create_baseline_model()
model.freeze_backbone()
features = model(exam_tensor)  # Forward pass
model.unfreeze_backbone()
trainable_params = model.get_trainable_params()
```

### Key Decisions Made

- **Backbone modification:** Remove final FC layer, keep ResNet18 feature extractor only
- **Max pooling location:** After ResNet features, before final classifier
- **Model structure:** Factory function returning instances (not separate classes)
- **Responsibility split:** Role 2 provides utilities, Role 4 controls training strategy
- **Checkpointing:** Role 4 handles all save/load logic

---

## Role 3: Comparative Models (Alternative Architectures)

**Owner:** [Team Member Name]  
**File:** `src/modules/comparative_models.py`

### Responsibilities

1. **Alternative Architecture Exploration**
  - Explore alternative CNN backbones while keeping other components constant
  - **Primary variants:**
    - ResNet50 (ImageNet pretrained) - deeper network, more capacity
    - ResNet18/50 (RadImageNet pretrained) - medical imaging specific weights
2. **Maintain Baseline Consistency**
  - Keep same as baseline: max pooling aggregation, weighted BCE loss, data preprocessing
  - **Only change:** the backbone architecture
  - Ensures fair comparison (isolates the variable)
3. **Code Reuse**
  - Import and use `MRNetBaseModel` base class from Role 2
  - Swap only the backbone network, reuse all other logic
  - Same freeze/unfreeze interface as baseline
4. **Model Factory Function**
  - Implement: `create_comparative_model(architecture, condition=None, plane=None)`
  - `architecture` options: 'resnet50', 'radimagenet_resnet18', 'radimagenet_resnet50'
  - Returns model with same interface as baseline

### Interface to Other Roles

**Inputs:**

- `MRNetBaseModel` base class from Role 2

**Outputs:**

- Factory function for creating comparative model instances
- Same freeze/unfreeze API as baseline

**Key Methods:**

```python
model = create_comparative_model(architecture='resnet50')
model.freeze_backbone()
# Same interface as baseline models
```

### Key Decisions Made

- **Scope:** Architecture exploration only (not loss functions, aggregation methods, etc.)
- **Architectures:** ResNet50 and RadImageNet variants (staying within CNN family)
- **Code pattern:** Unified factory function matching Role 2's interface
- **Shared infrastructure:** Reuses `MRNetBaseModel` base class

### Future Exploration (Optional)

If time permits, Role 3 may explore:

- Advanced slice aggregation (Attention mechanisms, Bidirectional LSTMs)
- Focal Loss (instead of weighted BCE)
- Other CNN architectures (EfficientNet, DenseNet)

---

## Role 4: Training Pipeline & Hyperparameter Tuning

**Owner:** [Team Member Name]  
**File:** `src/train.py`

### Responsibilities

1. **Training Orchestration for HPC**
  - Create modular `train_single.py` script that trains ONE model
  - Accept command-line arguments: `--condition`, `--plane`, `--architecture`
  - Create SLURM batch submission scripts (`submit_job.sh`, `submit_all.sh`)
  - Enable parallel training of multiple models via SLURM job queue
2. **Pilot Run & Time Estimation**
  - Run 1 model first (e.g., ACL-sagittal) to estimate training time
  - Use results to set appropriate SLURM walltime limits
  - Then submit remaining jobs
3. **Data Split Management**
  - Implement 80/20 split of `train/` folder → train/validation
  - Use validation for early stopping and hyperparameter tuning
  - Reserve provided `valid/` folder as held-out test set (for Role 5)
  - Ensure same random seed for reproducibility across all models
4. **Hyperparameter Tuning Strategy**
  - Tune on ONE representative model (e.g., ACL-sagittal baseline)
  - Hyperparameters to tune:
    - Learning rates (Phase 1 and Phase 2)
    - Number of epochs per phase
    - Optimizer choice (Adam, AdamW, SGD)
    - Potentially: weight decay, learning rate schedule
  - Apply best hyperparameters to all 9 models
5. **Two-Phase Training Strategy**
  - **Phase 1 (Freeze):**
    - Call `model.freeze_backbone()`
    - Train only custom layers (max pool + FC head)
    - Higher learning rate (e.g., 1e-3)
    - Fewer epochs (e.g., 10)
  - **Phase 2 (Fine-tune):**
    - Call `model.unfreeze_backbone()`
    - Train all layers end-to-end
    - Lower learning rate (e.g., 1e-4 or 1e-5)
    - More epochs (e.g., 20-30)
6. **Loss Function Setup**
  - Use `BCEWithLogitsLoss(pos_weight=dataset.pos_weight)` from Role 1
  - Different `pos_weight` for each condition (ACL, Meniscus, Abnormal)
7. **Experiment Tracking**
  - **File-based:** Save checkpoints, configs (YAML), metadata
  - **TensorBoard:** Log training/validation loss and metrics per epoch
  - Checkpoint structure:
    ```
    checkpoints/
    ├── {condition}_{plane}_{architecture}/
    │   ├── config.yaml
    │   ├── best_model.pth
    │   ├── final_model.pth
    │   └── training_log.csv
    ```
8. **Checkpointing Logic**
  - Save best model (based on validation AUC or loss)
  - Save final model (end of training)
  - Include metadata: condition, plane, architecture, hyperparameters, training time

### Interface to Other Roles

**Inputs:**

- Dataset instances from Role 1 (with `pos_weight`)
- Model factory functions from Role 2 and Role 3
- Freeze/unfreeze utilities from models

**Outputs:**

- Trained model checkpoints for all models
- Training logs and metrics
- Configuration files documenting hyperparameters used

**Key Scripts:**

```bash
# Train a single model
python train_single.py --condition ACL --plane axial --architecture baseline

# Submit all 9 baseline models to SLURM
bash submit_all.sh baseline

# Submit comparative models
bash submit_all.sh resnet50
```

### Key Decisions Made

- **Compute strategy:** SLURM batch jobs, 1 GPU per job, parallel execution
- **Hyperparameter tuning:** Tune once, apply to all (practical for course project)
- **Training approach:** Two-phase (freeze then fine-tune)
- **Tracking:** Hybrid (files + TensorBoard)
- **Responsibility:** Role 4 handles all save/load logic (not Role 2/3)

---

## Role 5: Evaluation & Metrics

**Owner:** [Team Member Name]  
**File:** `src/evaluation.py`

### Responsibilities

1. **Multi-Plane Prediction Fusion**
  - Load all 3 plane models for a given condition (e.g., ACL-axial, ACL-coronal, ACL-sagittal)
  - Generate predictions from each plane on test set
  - **Fusion method:** Simple averaging of probabilities
    - `final_prob = (prob_axial + prob_coronal + prob_sagittal) / 3`
2. **Comprehensive Metrics Suite**
  - Compute for each condition:
    - **AUC (Area Under ROC Curve)** - primary metric
    - **Sensitivity (Recall)** - true positive rate
    - **Specificity** - true negative rate
    - **F1-Score** - harmonic mean of precision and recall
  - All metrics reported with **95% confidence intervals** (bootstrap method)
3. **Classification Threshold**
  - Use **0.5** as default threshold for binary metrics (Sensitivity, Specificity, F1)
  - Always report AUC (threshold-independent)
  - Generate ROC and Precision-Recall curves (show all thresholds)
4. **Multi-Level Reporting**
  - **Level 1: Per-plane results**
    - Individual performance for each plane (e.g., ACL-axial: AUC 0.82)
    - Helps identify which planes are most informative
  - **Level 2: Fused results per condition**
    - Combined performance after 3-plane fusion (e.g., ACL-fused: AUC 0.87)
    - Primary results for final evaluation
5. **Model Comparison**
  - Side-by-side comparison tables: Baseline vs Comparative models
  - Report metrics with confidence intervals for uncertainty quantification
  - No formal statistical significance testing (confidence intervals provide visual comparison)
6. **Visualization**
  - Generate ROC curves (one per condition)
  - Generate Precision-Recall curves (especially informative for imbalanced classes)
  - Overlay baseline and comparative model curves for comparison
7. **Error Analysis**
  - Identify and log:
    - High-confidence correct predictions (True Positives/Negatives with p > 0.9 or p < 0.1)
    - Errors (False Positives, False Negatives)
    - Borderline cases (predictions near 0.5 threshold)
  - Pass error cases to Role 6 for explainability analysis

### Interface to Other Roles

**Inputs:**

- Trained model checkpoints from Role 4 (all 9 models per architecture)
- Test dataset instances from Role 1
- Model loading utilities

**Outputs:**

- Results tables (CSV/markdown) with metrics and confidence intervals
- ROC and PR curve figures
- Error analysis report
- List of example cases for Role 6 (correct, errors, borderline)

**Key Functions:**

```python
# Load and evaluate models
results = evaluate_models(condition='ACL', architecture='baseline', test_loader)

# Generate comparison report
comparison = compare_architectures(baseline_results, comparative_results)

# Get examples for explainability
examples = select_visualization_examples(predictions, labels, n_per_category=2)
```

### Key Decisions Made

- **Fusion strategy:** Simple averaging (equal weight to all planes)
- **Threshold:** 0.5 for binary classification, but emphasize AUC
- **Statistical rigor:** Confidence intervals, no formal hypothesis testing
- **Reporting:** Both per-plane and fused results
- **Evaluation set:** Held-out test set (provided `valid/` folder)

---

## Role 6: Explainability & Clinical Presentation

**Owner:** [Team Member Name]  
**File:** `src/explainability.py`

### Responsibilities

1. **Grad-CAM Implementation**
  - Implement Grad-CAM (Gradient-weighted Class Activation Mapping)
  - **Key insight:** Max pooling identifies which slice(s) drove the prediction
  - Generate heatmap for the **most important single slice** (baseline)
  - Optional: Extend to top-K slices (e.g., top 3-5) if time permits
2. **Example Selection**
  - Select **2 examples per category** per condition (~18 total visualizations)
  - Categories:
    - **High-confidence correct:** True Positives (p > 0.9), True Negatives (p < 0.1)
    - **Errors:** False Positives, False Negatives
    - **Borderline cases:** Predictions near threshold (0.4 < p < 0.6)
  - Coordinate with Role 5 for case selection
3. **Visualization Format**
  - **Baseline (Option A):** Basic heatmap overlay on MRI slice
  - **Report version (Option B):** Annotated with:
    - Probability score
    - Ground truth label and prediction
    - Brief anatomical context (e.g., "Model focuses on ACL region")
  - **Stretch goal (Option C):** Full clinical report format showing all 3 planes
4. **Validation of Explainability**
  - **Qualitative assessment:** Visual inspection of heatmaps
    - Do they highlight knee joint regions?
    - Avoid background/artifacts/image borders?
  - **Literature comparison:** 
    - Compare to anatomical references and known tear locations
    - Reference original MRNet paper's visualizations
    - Check alignment with expected anatomy (e.g., ACL visible in sagittal plane)
5. **Clinical Communication**
  - Generate clear, annotated figures suitable for report/presentation
  - Write qualitative analysis of model attention patterns
  - Discuss whether model focuses on anatomically relevant regions
  - Identify any concerning failure modes (e.g., attending to artifacts)

### Interface to Other Roles

**Inputs:**

- Best trained model checkpoints from Role 4
- Example cases from Role 5 (correct, errors, borderline)
- Model architecture from Role 2 (for hooking Grad-CAM)

**Outputs:**

- Grad-CAM heatmap visualizations (~18 annotated figures)
- Qualitative analysis document
- Figures formatted for report/presentation

**Key Functions:**

```python
# Generate Grad-CAM for a specific exam
heatmap, slice_idx = generate_gradcam(model, exam_tensor, target_class)

# Create annotated visualization
fig = create_clinical_visualization(
    exam_tensor, heatmap, slice_idx, 
    prediction, ground_truth, probability
)

# Validate attention patterns
validation_report = validate_heatmaps(heatmaps, anatomical_references)
```

### Key Decisions Made

- **Grad-CAM target:** Most important single slice from max pooling
- **Example selection:** 2 per category × 3 categories × 3 conditions = ~18 visualizations
- **Visualization stages:** Basic → Annotated → Clinical showcase (if time)
- **Validation approach:** Qualitative + literature comparison (no clinical expert required)

---

## Cross-Cutting Concerns & Dependencies

### 1. Interface Contracts

**Role 1 → Role 2/3/4:**

- Dataset class with consistent API
- `pos_weight` attribute for loss function

**Role 2 → Role 3:**

- `MRNetBaseModel` base class
- Freeze/unfreeze interface

**Role 2/3 → Role 4:**

- Factory functions: `create_baseline_model()`, `create_comparative_model()`
- Training utilities

**Role 4 → Role 5:**

- Trained model checkpoints
- Metadata and configuration files

**Role 5 → Role 6:**

- Error analysis and example selection
- Performance metrics for context

### 2. Timeline Dependencies

**Sequential dependencies:**

1. Role 1 must complete dataset implementation before any training
2. Role 2 must implement `MRNetBaseModel` before Role 3 can build on it
3. Role 4 needs Role 1 (data) and Role 2 (baseline models) to start training
4. Role 5 needs trained models from Role 4
5. Role 6 needs results from Role 5 for example selection

**Parallelizable work:**

- Role 2 and Role 1 can work in parallel (once interfaces are agreed)
- Role 3 can start once Role 2's base class is implemented
- Role 5 and Role 6 can work on implementation while Role 4 trains models

### 3. Shared Resources

**Configuration Management:**

- Centralized config files (YAML) for hyperparameters
- Consistent random seeds across all experiments
- Documented in Role 4's checkpoint directories

**Data Paths:**

- Agreed-upon directory structure for data, checkpoints, results
- Environment variables or config file for paths

**Code Standards:**

- Consistent function/class naming conventions
- Type hints where possible
- Docstrings for public functions
- Unit tests for critical components (optional but recommended)

### 4. Integration Testing

**Key integration points to test:**

- Role 1 → Role 4: Dataset loads correctly in training loop
- Role 2 → Role 4: Models train without errors, checkpoints save/load
- Role 4 → Role 5: Checkpoints load correctly for evaluation
- Role 5 → Role 6: Example selection pipeline works

---

## Glossary of Key Terms

See [CONTEXT.md](CONTEXT.md) for detailed definitions of domain terminology used throughout this project.