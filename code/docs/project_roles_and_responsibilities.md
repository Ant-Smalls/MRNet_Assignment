# MRNet Project: Roles and Responsibilities

## Project Context and Overview

This document outlines the division of work for the MRNet deep learning classification project across 6 team members. The project builds on the architectural decisions documented in the [Initial Research Plan](initial_research_plan.md), which established:

- **9 independent binary models** (3 conditions × 3 planes: ACL, Meniscus, Abnormal across Axial, Coronal, Sagittal)
- **Max pooling** for slice aggregation (baseline)
- **Transfer learning** from ImageNet pre-trained ResNet18
- **Weighted loss functions** to handle class imbalance
- **Medical imaging-specific augmentations** (no horizontal flip, contrast/brightness jitter)

### Novel Methodological Extensions

This project extends the baseline approach with:

1. **Model-Guided Slice Selection:** Two-phase training (all slices → identify top-5 → retrain on filtered slices)
2. **Medical Imaging Pretraining Comparison:** RadImageNet vs ImageNet weights
3. **Clinical Triage Interface:** Confidence-based prioritization with multi-plane explainability

### Total Model Count

- **Per Architecture:** 18 models (9 all-slices + 9 filtered)
- **Architectures:** 2 (ResNet18-ImageNet baseline + ResNet50-RadImageNet comparative)
- **Total:** 36 models across all experiments
- **Primary Focus:** ACL condition (12 models) for main analysis and triage tool

### Project File Structure

```
code/
├── src/
│   ├── modules/
│   │   ├── data_preprocessing_transformation.py  (Role 1)
│   │   │   └── NEW: FilteredDataset, extract_top_k_slices()
│   │   ├── baseline_models.py                    (Role 2)
│   │   │   └── NEW: forward_with_slice_tracking()
│   │   └── comparative_models.py                 (Role 3)
│   ├── train.py                                   (Role 4)
│   │   └── NEW: Multi-phase training (all/filtered)
│   ├── evaluation.py                              (Role 5)
│   │   └── NEW: Detailed prediction export for triage
│   └── explainability.py                          (Role 6)
│       └── NEW: Clinical triage interface
├── metadata/
│   └── slice_selections/                          (NEW)
│       ├── acl_axial_baseline_all.json
│       ├── acl_axial_baseline_filtered.json
│       └── ...
├── checkpoints/
│   ├── {condition}_{plane}_{arch}_all_slices/    (NEW structure)
│   └── {condition}_{plane}_{arch}_filtered/      (NEW structure)
└── docs/
    ├── initial_research_plan.md
    ├── project_roles_and_responsibilities.md
    └── CONTEXT.md
```

### Key Cross-Cutting Decisions

- **Compute Environment:** SLURM HPC with batch job submission, 1 GPU per job
- **Data Split Strategy:** 80/20 split of `train/` folder (train/validation), held-out `valid/` folder as test set
- **Experiment Tracking:** Hybrid approach - file-based checkpoints + TensorBoard for training curves
- **Model Count:** 36 models total (18 per architecture: 9 all-slices + 9 filtered)
- **Primary Focus:** ACL condition for triage tool demonstration
- **Shared Code:** `MRNetBaseModel` base class implemented by Role 2, reused by Role 3
- **Training Workflow:** Sequential phases (all-slices → slice extraction → filtered retraining)

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
5. **Model-Guided Slice Selection (NEW)**
  - Implement `extract_top_k_slices(model, dataset, k=5)` utility function
  - Runs inference on dataset and tracks which slices contribute to max pooling
  - Returns dictionary: `{exam_id: [top_k_slice_indices]}`
  - Saves slice selection metadata to JSON files per condition-plane-architecture
6. **Filtered Dataset Implementation (NEW)**
  - Implement `FilteredDataset` class with same interface as standard `Dataset`
  - Loads only pre-selected slices based on metadata file
  - Constructor: `FilteredDataset(condition, plane, split, slice_metadata_path)`
  - Enables retraining on reduced, informative slice sets

### Interface to Other Roles

**Outputs:**

- Dataset instances for each condition-plane-split combination
- `pos_weight` attribute accessible to Role 4
- Slice selection metadata files (JSON) for filtered training
- `FilteredDataset` class for retraining on selected slices

**Key Methods:**

```python
# Standard dataset
dataset = Dataset(condition='ACL', plane='axial', split='train')
exam, label = dataset[idx]  # Returns (num_slices, H, W) tensor and label
pos_weight = dataset.pos_weight  # For loss function

# Slice selection utility (uses Role 2's slice tracking)
slice_selections = extract_top_k_slices(model, dataset, k=5)
# Returns: {exam_id: [slice_indices]}

# Filtered dataset for retraining
filtered_dataset = FilteredDataset(
    condition='ACL', plane='axial', split='train',
    slice_metadata_path='slices/acl_axial_baseline.json'
)
```

### Key Decisions Made

- **Dataset structure:** Single parameterized class (not separate classes per condition)
- **Slice handling:** Variable-length tensors (no padding/truncation)
- **Batching:** Batch size = 1
- **Augmentation timing:** On-the-fly during training
- **Weight computation:** Calculated in dataset `__init__`, per condition
- **Slice selection scope:** Per-exam (dynamic), independent per condition-plane-architecture
- **Top-K value:** 5 slices retained per exam
- **Metadata format:** JSON files mapping exam_id to slice indices

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
4. **Slice Tracking for Interpretability (NEW)**
  - Implement `forward_with_slice_tracking(x)` method
  - Returns both prediction AND which slices contributed to max pooling
  - Output format: `(logits, slice_indices)` where slice_indices has shape (512,)
  - Used by Role 1 for slice selection and Role 6 for explainability
5. **Model Factory Function**
  - Implement: `create_baseline_model(condition=None, plane=None)`
  - Returns model instance with baseline ResNet18 architecture
  - All 9 models share identical architecture (just trained on different data)

### Interface to Other Roles

**Outputs:**

- `MRNetBaseModel` base class (used by Role 3)
- Factory function for creating model instances
- Freeze/unfreeze utilities (used by Role 4)
- Slice tracking capability (used by Role 1 and Role 6)

**Key Methods:**

```python
model = create_baseline_model()
model.freeze_backbone()
logits = model(exam_tensor)  # Standard forward pass

# For slice selection and explainability
logits, slice_indices = model.forward_with_slice_tracking(exam_tensor)
# slice_indices: tensor of shape (512,) indicating which slice contributed 
# the max activation for each of the 512 feature dimensions

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
  - **Primary variant:**
    - **ResNet50 (RadImageNet pretrained)** - medical imaging specific weights, deeper network
  - **Secondary variants (if time permits):**
    - ResNet18 (RadImageNet) - same depth as baseline, medical weights
    - Other medical imaging pretrained architectures
2. **Maintain Baseline Consistency**
  - Keep same as baseline: max pooling aggregation, weighted BCE loss, data preprocessing
  - **Only change:** the backbone architecture and pretrained weights
  - Ensures fair comparison (isolates the variable)
3. **Code Reuse**
  - Import and use `MRNetBaseModel` base class from Role 2
  - Swap only the backbone network, reuse all other logic (including slice tracking)
  - Same freeze/unfreeze interface as baseline
4. **Model Factory Function**
  - Implement: `create_comparative_model(architecture, condition=None, plane=None)`
  - `architecture` options: 'radimagenet_resnet50' (primary), 'radimagenet_resnet18' (optional)
  - Returns model with same interface as baseline (including `forward_with_slice_tracking()`)
5. **RadImageNet Integration**
  - Load RadImageNet pretrained weights (requires separate download/setup)
  - Handle potential differences in architecture initialization
  - Ensure compatibility with slice tracking mechanism

### Interface to Other Roles

**Inputs:**

- `MRNetBaseModel` base class from Role 2 (including slice tracking capability)
- RadImageNet pretrained weights (external resource)

**Outputs:**

- Factory function for creating comparative model instances
- Same freeze/unfreeze API as baseline
- Same slice tracking API (`forward_with_slice_tracking()`)

**Key Methods:**

```python
model = create_comparative_model(architecture='radimagenet_resnet50')
model.freeze_backbone()

# Standard forward pass
logits = model(exam_tensor)

# Slice tracking (inherited from MRNetBaseModel)
logits, slice_indices = model.forward_with_slice_tracking(exam_tensor)
```

### Key Decisions Made

- **Scope:** Architecture exploration only (not loss functions, aggregation methods, etc.)
- **Primary architecture:** ResNet50 with RadImageNet pretraining (medical imaging focus)
- **Code pattern:** Unified factory function matching Role 2's interface
- **Shared infrastructure:** Reuses `MRNetBaseModel` base class with slice tracking
- **Comparison focus:** Medical imaging weights vs general ImageNet weights (not CNN vs Transformer)

### Future Exploration (Optional)

If time permits and primary experiments are complete, Role 3 may explore:

- ResNet18 (RadImageNet) - isolate effect of medical pretraining at same depth
- Vision Transformers (DINOv2, ViT) - would require different slice tracking mechanism
- Other medical imaging architectures from RadImageNet or similar initiatives

**Note:** Vision Transformers are currently out of scope due to different attention-based aggregation that would complicate slice selection comparisons.

---

## Role 4: Training Pipeline & Hyperparameter Tuning

**Owner:** [Team Member Name]  
**File:** `src/train.py`

### Responsibilities

1. **Training Orchestration for HPC**
  - Create modular `train_single.py` script that trains ONE model
  - Accept command-line arguments: `--condition`, `--plane`, `--architecture`, `--use_filtered` (NEW)
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
5. **Two-Phase Transfer Learning Strategy**
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
6. **Multi-Phase Training for Slice Selection (NEW)**
  - **Training Phase A: All Slices**
    - Train each model on complete dataset (all slices)
    - Save checkpoints as `{condition}_{plane}_{architecture}_all_slices/`
  - **Slice Extraction (Coordination with Role 1):**
    - After Phase A completes, Role 1 uses trained models to extract top-5 slices
    - Role 1 generates slice metadata files and `FilteredDataset` instances
  - **Training Phase B: Filtered Slices**
    - Retrain same architectures on filtered datasets
    - Save checkpoints as `{condition}_{plane}_{architecture}_filtered/`
    - Use same hyperparameters as Phase A for fair comparison
7. **Loss Function Setup**
  - Use `BCEWithLogitsLoss(pos_weight=dataset.pos_weight)` from Role 1
  - Different `pos_weight` for each condition (ACL, Meniscus, Abnormal)
8. **Experiment Tracking**
  - **File-based:** Save checkpoints, configs (YAML), metadata
  - **TensorBoard:** Log training/validation loss and metrics per epoch
  - Checkpoint structure:
    ```
    checkpoints/
    ├── {condition}_{plane}_{architecture}_all_slices/
    │   ├── config.yaml
    │   ├── best_model.pth
    │   ├── final_model.pth
    │   └── training_log.csv
    ├── {condition}_{plane}_{architecture}_filtered/
    │   ├── config.yaml
    │   ├── best_model.pth
    │   ├── final_model.pth
    │   └── training_log.csv
    ```
9. **Checkpointing Logic**
  - Save best model (based on validation AUC or loss)
  - Save final model (end of training)
  - Include metadata: condition, plane, architecture, slice_mode (all/filtered), hyperparameters, training time

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
# Train a single model (all slices)
python train_single.py --condition ACL --plane axial --architecture baseline

# Train on filtered slices (after Role 1 extracts slice selections)
python train_single.py --condition ACL --plane axial --architecture baseline --use_filtered

# Submit all 9 baseline models to SLURM (Phase A - all slices)
bash submit_all.sh baseline all_slices

# Submit filtered training jobs (Phase B - after slice extraction)
bash submit_all.sh baseline filtered

# Submit comparative models
bash submit_all.sh radimagenet all_slices
bash submit_all.sh radimagenet filtered
```

### Key Decisions Made

- **Compute strategy:** SLURM batch jobs, 1 GPU per job, parallel execution
- **Hyperparameter tuning:** Tune once, apply to all (practical for course project)
- **Training approach:** Two-phase (freeze then fine-tune)
- **Tracking:** Hybrid (files + TensorBoard)
- **Responsibility:** Role 4 handles all save/load logic (not Role 2/3)
- **Slice selection workflow:** Train all-slices first, then filtered (sequential dependency)
- **Model count per architecture:** 18 models total (9 all-slices + 9 filtered) × 2 architectures = 36 models
- **Primary focus:** ACL condition (3 planes × 2 slice modes × 2 architectures = 12 models) for main analysis

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
  - **Level 3: Slice mode comparison (NEW)**
    - Compare all-slices vs filtered-slices performance
    - Report for both baseline and comparative architectures
5. **Model Comparison**
  - Side-by-side comparison tables: 
    - Baseline vs Comparative models
    - All-slices vs Filtered-slices (NEW)
  - Report metrics with confidence intervals for uncertainty quantification
  - No formal statistical significance testing (confidence intervals provide visual comparison)
6. **Visualization**
  - Generate ROC curves (one per condition)
  - Generate Precision-Recall curves (especially informative for imbalanced classes)
  - Overlay baseline and comparative model curves for comparison
7. **Error Analysis & Confidence Export (ENHANCED)**
  - Identify and log:
    - High-confidence correct predictions (True Positives/Negatives with p > 0.9 or p < 0.1)
    - Errors (False Positives, False Negatives)
    - Borderline cases (predictions near 0.5 threshold)
  - **NEW: Export detailed predictions for Role 6:**
    - Per-exam predictions with both per-plane AND fused confidences
    - Format: JSON file with exam_id, per-plane scores, fused score, ground truth
    - Enables triage tool to display multi-plane confidence breakdown

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
- **NEW: Detailed prediction export for triage tool:**
  ```json
  {
    "exam_id": "0042",
    "condition": "ACL",
    "per_plane_predictions": {
      "axial": {"confidence": 0.72, "model_path": "..."},
      "coronal": {"confidence": 0.91, "model_path": "..."},
      "sagittal": {"confidence": 0.98, "model_path": "..."}
    },
    "fused_confidence": 0.87,
    "ground_truth": 1,
    "prediction": 1
  }
  ```

**Key Functions:**

```python
# Load and evaluate models (both slice modes)
results_all = evaluate_models(condition='ACL', architecture='baseline', 
                                slice_mode='all_slices', test_loader)
results_filtered = evaluate_models(condition='ACL', architecture='baseline',
                                     slice_mode='filtered', test_loader)

# Generate comparison report
comparison = compare_architectures(baseline_results, comparative_results)
slice_comparison = compare_slice_modes(all_slices_results, filtered_results)

# Get examples for explainability
examples = select_visualization_examples(predictions, labels, n_per_category=2)

# Export predictions for triage tool (Role 6)
export_predictions_for_triage(predictions, output_path='predictions/acl_triage.json')
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
6. **Novel Clinical Triage Interface (NEW)**
  - **Confidence-Based Ranking:**
    - Load prediction export from Role 5
    - Sort test cases by fused confidence (high to low)
    - Identify priority cases for review
  - **Multi-Plane Visualization:**
    - For each prioritized case, load all 3 plane models
    - Use Role 2's `forward_with_slice_tracking()` to identify most informative slice per plane
    - Generate Grad-CAM overlay for the informative slice from each plane
  - **Interface Components:**
    - Display case grid sorted by priority (confidence score)
    - For each case, show:
      - Fused confidence score and per-plane confidences
      - Three thumbnails (axial, coronal, sagittal) with Grad-CAM overlays
      - Most informative slice index and total slices per plane
      - Ground truth label for validation
  - **Implementation Format:**
    - Jupyter notebook with interactive widgets OR
    - Simple HTML interface with thumbnail grid
  - **Scope:**
    - Primary focus: Single condition (ACL) as proof of concept
    - Display top-N cases (e.g., N=20-50 highest confidence predictions)
    - Extensible to all 3 conditions if time permits

### Interface to Other Roles

**Inputs:**

- Best trained model checkpoints from Role 4 (all 3 planes per condition)
- Example cases from Role 5 (correct, errors, borderline)
- Model architecture from Role 2 (for hooking Grad-CAM and slice tracking)
- **NEW:** Detailed prediction export from Role 5 (per-plane + fused confidences)

**Outputs:**

- Grad-CAM heatmap visualizations (~18 annotated figures for traditional examples)
- Qualitative analysis document
- Figures formatted for report/presentation
- **NEW: Interactive triage interface**
  - Jupyter notebook or HTML page
  - Top-N prioritized cases with 3-plane visualization
  - Grad-CAM overlays on most informative slices
  - Demonstration of clinical workflow integration

**Key Functions:**

```python
# Traditional Grad-CAM for selected examples
heatmap, slice_idx = generate_gradcam(model, exam_tensor, target_class)

# Create annotated visualization
fig = create_clinical_visualization(
    exam_tensor, heatmap, slice_idx, 
    prediction, ground_truth, probability
)

# Validate attention patterns
validation_report = validate_heatmaps(heatmaps, anatomical_references)

# NEW: Triage interface functions
def extract_informative_slices_multiplane(exam_id, plane_models, dataset):
    """
    For a single exam, load data for all 3 planes and extract most informative slice per plane.
    Returns: {plane_name: (slice_idx, total_slices)}
    """
    
def generate_gradcam_multiplane(exam_id, plane_models, slice_info, dataset):
    """
    Generate Grad-CAM overlays for the most informative slice from each plane.
    Returns: {plane_name: gradcam_image}
    """

def create_triage_interface(predictions_json, models, dataset, top_n=50):
    """
    Main function to generate the triage interface.
    Loads predictions, generates multi-plane visualizations, creates sorted display.
    """
```

### Key Decisions Made

- **Grad-CAM target:** Most important single slice from max pooling
- **Example selection:** 2 per category × 3 categories × 3 conditions = ~18 visualizations
- **Visualization stages:** Basic → Annotated → Clinical showcase (if time)
- **Validation approach:** Qualitative + literature comparison (no clinical expert required)
- **Triage interface scope:** Single condition (ACL) for primary deliverable, multi-plane visualization
- **Triage display format:** Top-N prioritized cases (N=20-50) with confidence scores
- **Novel contribution:** Combines model confidence, slice selection, and explainability for clinical workflow

---

## Cross-Cutting Concerns & Dependencies

### 1. Interface Contracts

**Role 1 → Role 2/3/4:**

- Dataset class with consistent API
- `pos_weight` attribute for loss function
- **NEW:** `FilteredDataset` class for retraining
- **NEW:** `extract_top_k_slices()` utility (uses Role 2's tracking)

**Role 2 → Role 1:**

- **NEW:** `forward_with_slice_tracking()` method for slice selection

**Role 2 → Role 3:**

- `MRNetBaseModel` base class
- Freeze/unfreeze interface
- **NEW:** Slice tracking capability inherited by comparative models

**Role 2/3 → Role 4:**

- Factory functions: `create_baseline_model()`, `create_comparative_model()`
- Training utilities
- **NEW:** Slice tracking support for both architectures

**Role 4 → Role 1:**

- **NEW:** After Phase A training completes, signals Role 1 to extract slices
- **NEW:** Waits for slice metadata files before starting Phase B training

**Role 4 → Role 5:**

- Trained model checkpoints (all-slices and filtered versions)
- Metadata and configuration files
- **NEW:** Separate checkpoint directories per slice mode

**Role 5 → Role 6:**

- Error analysis and example selection
- Performance metrics for context
- **NEW:** Detailed per-plane + fused confidence exports (JSON)

**Role 1/2 → Role 6:**

- **NEW:** Slice tracking utilities for triage interface
- Dataset loaders for visualization

### 2. Timeline Dependencies

**Sequential dependencies:**

1. Role 1 must complete standard dataset implementation before any training
2. Role 2 must implement `MRNetBaseModel` with slice tracking before Role 3 can build on it
3. **Phase A Training:** Role 4 needs Role 1 (data) and Role 2 (baseline models) to train all-slices models
4. **Slice Extraction:** Role 1 needs trained models from Role 4 Phase A to extract top-K slices
5. **Phase B Training:** Role 4 needs slice metadata from Role 1 to train filtered models
6. Role 5 needs trained models from Role 4 (both phases) for evaluation
7. Role 6 needs results from Role 5 for example selection and triage interface

**Parallelizable work:**

- Role 2 and Role 1 can work on standard implementations in parallel (once interfaces are agreed)
- Role 3 can start once Role 2's base class is implemented
- Role 5 and Role 6 can work on implementation scaffolding while Role 4 trains models
- **NEW:** Phase A training can run all models in parallel via SLURM
- **NEW:** Phase B training can run all filtered models in parallel after slice extraction completes

### 3. Critical Path for Novel Methodology

The slice selection workflow creates these key milestones:

**Milestone 1: Standard Infrastructure** (Roles 1, 2, 3)

- Dataset, models, training pipeline operational
- No blockers for parallel development

**Milestone 2: Phase A Training Complete** (Role 4)

- All 18 all-slices models trained (9 baseline + 9 comparative)
- Checkpoints saved and validated
- **Blocks:** Slice extraction cannot begin until this completes

**Milestone 3: Slice Selection Complete** (Role 1)

- Top-K slices identified for all 18 models
- Metadata files generated and saved
- `FilteredDataset` tested and working
- **Blocks:** Phase B training cannot begin until this completes

**Milestone 4: Phase B Training Complete** (Role 4)

- All 18 filtered models trained
- Both all-slices and filtered checkpoints available
- **Unblocks:** Full evaluation and comparison

**Milestone 5: Evaluation & Triage** (Roles 5, 6)

- All comparisons complete (all vs filtered, baseline vs comparative)
- Triage interface operational for best-performing model
- Final deliverable

### 4. Shared Resources

**Configuration Management:**

- Centralized config files (YAML) for hyperparameters
- Consistent random seeds across all experiments
- Documented in Role 4's checkpoint directories
- **NEW:** Separate configs for all-slices and filtered training

**Data Paths:**

- Agreed-upon directory structure for data, checkpoints, results
- Environment variables or config file for paths
- **NEW:** `metadata/slice_selections/` directory for slice metadata

**Code Standards:**

- Consistent function/class naming conventions
- Type hints where possible
- Docstrings for public functions
- Unit tests for critical components (optional but recommended)
- **NEW:** Document slice selection algorithm and rationale in code comments

### 5. Integration Testing

**Key integration points to test:**

- Role 1 → Role 4: Dataset loads correctly in training loop (standard and filtered)
- Role 2 → Role 4: Models train without errors, checkpoints save/load
- **NEW:** Role 2 → Role 1: Slice tracking produces valid slice indices
- **NEW:** Role 1 → Role 4: FilteredDataset loads correct slices, handles variable lengths
- Role 4 → Role 5: Checkpoints load correctly for evaluation (both slice modes)
- **NEW:** Role 5 → Role 6: Prediction JSON format matches triage interface expectations
- **NEW:** Role 6: Multi-plane visualization loads 3 models and generates Grad-CAMs successfully

---

## Glossary of Key Terms

See [CONTEXT.md](CONTEXT.md) for detailed definitions of domain terminology used throughout this project.