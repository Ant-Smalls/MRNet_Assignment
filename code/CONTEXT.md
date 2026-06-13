# MRNet Project Context & Terminology

This document defines the domain language and key terminology for the MRNet knee MRI classification project. All team members should use this shared vocabulary to ensure clear communication.

---

## Domain Concepts

### Medical Imaging Terms

**Exam**  
A complete MRI scan of a single patient's knee. Each exam consists of three imaging planes (axial, coronal, sagittal), where each plane contains multiple 2D slices.

**Slice**  
A single 2D MRI image from one plane of an exam. Slices are stacked to form a 3D volume. The number of slices varies per exam and per plane (typically 20-40 slices).

**Plane**  
One of three orthogonal views of the knee anatomy:
- **Axial:** Top-down view (horizontal cross-sections)
- **Coronal:** Front-back view (frontal cross-sections)
- **Sagittal:** Side view (lateral cross-sections)

Each plane provides different perspectives on anatomical structures. For example, ACL tears are often most visible in the sagittal plane.

**Condition**  
The clinical classification task. Three conditions in this project:
- **ACL:** Anterior Cruciate Ligament tear detection
- **Meniscus:** Meniscal tear detection (medial or lateral meniscus)
- **Abnormal:** General abnormality detection (any pathology present)

---

## Model Architecture Terms

**Backbone**  
The feature extraction component of the model, typically a pre-trained CNN (e.g., ResNet18, ResNet50). The backbone processes 2D slices and outputs high-dimensional feature vectors. Also called "feature extractor."

**Feature Extractor**  
Synonym for backbone. The convolutional layers of a pre-trained model that extract visual features from images, before the final classification layer.

**Max Pooling Aggregation**  
The method used to combine features from multiple slices into a single representation. For an exam with S slices, the model extracts features `(S, 512)` and applies max pooling across the slice dimension to get `(512,)`. This answers: "What is the strongest evidence of a tear present in ANY slice?"

**Slice Tracking**  
A mechanism to monitor which specific slices contributed to the max pooling operation. During forward pass, the model tracks the slice index that provided the maximum activation for each feature dimension. Returns a tensor of shape `(512,)` indicating which slice was "most important" for each feature. Used for both interpretability (identifying informative slices) and slice selection (filtering dataset).

**Logit**  
The raw output of the model before applying sigmoid activation. For binary classification, the model outputs a single logit value. Applying sigmoid converts it to a probability in [0, 1].

**Head / Classification Head**  
The final layers of the model that convert features to predictions. In this project, typically a fully connected (FC) layer that maps from feature dimension (512) to a single binary output (1 logit).

---

## Training & Transfer Learning Terms

**Freeze / Frozen Backbone**  
Setting `requires_grad = False` for all parameters in the backbone, preventing them from being updated during training. Used in Phase 1 of two-phase training to preserve pre-trained weights while training only the custom classification head.

**Unfreeze / Fine-tune**  
Setting `requires_grad = True` to allow backbone parameters to be updated. Used in Phase 2 of two-phase training to adapt pre-trained features to the medical imaging domain.

**Two-Phase Training (Transfer Learning)**  
A transfer learning strategy:
1. **Phase 1:** Freeze backbone, train only custom layers (faster, prevents destroying pre-trained features)
2. **Phase 2:** Unfreeze backbone, fine-tune entire model with lower learning rate (adapts features to task)

**Multi-Phase Training (Slice Selection)**  
A novel training workflow that uses model feedback to filter the dataset:
1. **Phase A (All-Slices Training):** Train models on complete MRI volumes (all slices per exam)
2. **Slice Extraction:** Use trained models to identify top-K most informative slices per exam
3. **Phase B (Filtered Training):** Retrain same architectures on filtered datasets containing only selected slices
4. **Comparison:** Evaluate performance of all-slices vs filtered-slices models

**All-Slices Training / All-Slices Model**  
Training mode where each exam uses its complete set of MRI slices (typically 20-40 slices). Baseline approach that includes all available data without filtering.

**Filtered Training / Filtered Model**  
Training mode where each exam uses only a subset of pre-selected "most informative" slices (e.g., top-5). Slices are selected based on their contribution to max pooling in a previously trained all-slices model.

**pos_weight**  
A parameter of `BCEWithLogitsLoss` that upweights the positive class to handle class imbalance. Computed as `num_negative / num_positive`. For example, if ACL tears are 20% of the dataset (200 positive, 800 negative), `pos_weight = 800/200 = 4.0`.

**Checkpoint**  
A saved model state including weights, optimizer state, and training metadata. Typically save:
- **Best checkpoint:** Model with best validation performance (for evaluation)
- **Final checkpoint:** Model at end of training (for analysis)

**Pilot Run**  
Training a single model first to estimate training time, memory requirements, and validate the training pipeline before submitting all jobs. Essential for batch job systems like SLURM.

---

## Data & Preprocessing Terms

**Augmentation**  
Random transformations applied to training images to increase dataset diversity and reduce overfitting. In this project:
- **Included:** Contrast/brightness jittering (simulates different MRI scanners)
- **Excluded:** Horizontal flips (would incorrectly mirror medial/lateral anatomy)

**On-the-fly Augmentation**  
Augmentations applied during training when each batch is loaded, not pre-computed and saved to disk. Each epoch sees different random augmentations of the same images.

**Channel Stacking / 1→3 Channel Conversion**  
Repeating a single-channel grayscale image 3 times to create a 3-channel RGB-like tensor. Required to use ImageNet pre-trained models, which expect 3-channel inputs. Example: `(H, W)` → `(3, H, W)` by repeating along the channel dimension.

**Train/Val/Test Split**  
- **Train (80% of `train/` folder):** Used for model training (backpropagation, weight updates)
- **Validation (20% of `train/` folder):** Used for early stopping, hyperparameter tuning, and model selection
- **Test (provided `valid/` folder):** Held-out set for final evaluation only, never used during training or tuning

**Filtered Dataset**  
A dataset variant that loads only pre-selected slices for each exam, based on slice selection metadata. Constructed after all-slices training by identifying the top-K most informative slices per exam. Enables retraining on reduced, focused data.

**Slice Selection Metadata**  
JSON files mapping exam IDs to selected slice indices. Format: `{exam_id: [slice_indices]}`. Stored per condition-plane-architecture combination. Example: `{"0001": [5, 12, 18, 23, 27]}` indicates exam 0001 should use slices at indices 5, 12, 18, 23, and 27.

**Top-K Slices / Informative Slices**  
The K slices (typically K=5) that contributed most frequently to max pooling activations during inference. These are the slices the model relied on most heavily for its prediction. Selected dynamically per exam based on slice tracking.

---

## Evaluation & Metrics Terms

**Prediction Fusion / Multi-Plane Fusion**  
Combining predictions from models trained on different planes (axial, coronal, sagittal) into a single prediction. In this project, using simple averaging: `final_prob = (prob_axial + prob_coronal + prob_sagittal) / 3`.

**Per-Plane Evaluation**  
Reporting metrics for individual plane models separately (e.g., ACL-axial: AUC 0.82, ACL-coronal: AUC 0.84, ACL-sagittal: AUC 0.86). Helps identify which planes are most informative for each condition.

**Fused Evaluation**  
Reporting metrics after combining predictions from all 3 planes (e.g., ACL-fused: AUC 0.87). This is the final model performance metric.

**Slice Mode Comparison**  
Evaluating and comparing models trained with different slice selection strategies: all-slices (complete volumes) vs filtered-slices (top-K only). Reported alongside architecture comparisons (baseline vs comparative). Helps answer: "Does model-guided slice selection improve performance?"

**AUC (Area Under ROC Curve)**  
Primary evaluation metric. Measures the model's ability to rank positive cases higher than negative cases, independent of threshold choice. Perfect classifier: AUC = 1.0, random classifier: AUC = 0.5.

**Sensitivity / Recall / True Positive Rate**  
Proportion of actual positives correctly identified by the model. Critical in medical imaging where missing a tear (false negative) has high cost. Formula: `TP / (TP + FN)`.

**Specificity / True Negative Rate**  
Proportion of actual negatives correctly identified by the model. Formula: `TN / (TN + FP)`.

**Confidence Interval (95% CI)**  
A range estimating the uncertainty of a metric. Example: AUC = 0.85 [0.82-0.88] means we're 95% confident the true AUC lies between 0.82 and 0.88. Computed using bootstrap resampling.

**ROC Curve (Receiver Operating Characteristic)**  
A plot of True Positive Rate (Sensitivity) vs. False Positive Rate (1 - Specificity) across all possible classification thresholds. The area under this curve is the AUC metric.

**PR Curve (Precision-Recall Curve)**  
A plot of Precision vs. Recall across all thresholds. Particularly informative for imbalanced datasets like ACL tears, where PR curves better reflect performance on the minority class.

**Threshold**  
The probability cutoff for converting model outputs to binary predictions. Default is 0.5: predict positive if probability ≥ 0.5, negative otherwise. Choice of threshold affects Sensitivity, Specificity, and F1-Score, but not AUC.

---

## Explainability Terms

**Grad-CAM (Gradient-weighted Class Activation Mapping)**  
A visualization technique that highlights which regions of an input image most influenced the model's prediction. Generates a heatmap overlay showing areas of high importance (typically in red/warm colors).

**Saliency Map / Attention Map**  
Synonym for Grad-CAM heatmap. Shows where the model "looks" when making a decision.

**Max-Pooled Slice**  
The specific slice (or slices) that contributed most to the final prediction after max pooling. Since max pooling selects maximum feature activations across slices, we can track which slice(s) provided those maximums. These are the most "important" slices for Grad-CAM visualization.

**Clinical Interpretability**  
The degree to which model predictions and explanations are understandable and trustworthy to medical professionals. Validated by checking if Grad-CAM heatmaps focus on anatomically relevant regions (e.g., ACL region for ACL tear predictions, not background artifacts).

**Anatomical Validation**  
Confirming that model attention (via Grad-CAM) aligns with known anatomical structures and expected pathology locations based on medical literature and domain knowledge.

**Triage Interface / Clinical Triage Tool**  
A visualization system that prioritizes MRI cases for radiologist review based on model confidence. Displays cases in ranked order (highest confidence first) with multi-plane Grad-CAM overlays and informative slice identification. Simulates a clinical workflow where high-confidence predictions receive urgent attention.

**Multi-Plane Visualization**  
Displaying predictions and explanations from all three imaging planes (axial, coronal, sagittal) simultaneously for a single exam. Shows per-plane confidence scores, the fused confidence, and Grad-CAM overlays on the most informative slice from each plane. Provides comprehensive view of model reasoning.

**Priority Ranking / Confidence-Based Ranking**  
Sorting test cases by model confidence (fused probability) from highest to lowest. High-confidence positive predictions (e.g., p > 0.9) are flagged as priority cases requiring immediate review. Used in the triage interface to optimize radiologist workflow.

---

## Technical & Workflow Terms

**Factory Function**  
A function that creates and returns model instances with consistent configuration. Example: `create_baseline_model(condition, plane)` returns a configured ResNet18 model. Enables architecture-agnostic code in the training pipeline.

**SLURM (Simple Linux Utility for Resource Management)**  
The batch job scheduling system used on the Sonic HPC cluster. Jobs are submitted via `sbatch` command with resource specifications (GPUs, CPUs, walltime).

**Batch Job**  
A computational task submitted to a cluster scheduler (SLURM) that runs asynchronously. For this project, each batch job trains one model (one condition-plane-architecture combination).

**Walltime**  
The maximum time a batch job is allowed to run before being terminated. Must be estimated and specified when submitting SLURM jobs.

**Experiment Tracking**  
Systematic recording of hyperparameters, metrics, and artifacts for each training run. In this project, using file-based checkpoints + TensorBoard for visualization.

**TensorBoard**  
A visualization tool for monitoring training progress in real-time. Logs training/validation loss, metrics, and other scalars over epochs.

---

## Architecture-Specific Terms

**Baseline Model / Baseline Architecture**  
The reference model used as a performance benchmark. In this project: ResNet18 pre-trained on ImageNet, with max pooling aggregation. All improvements are measured against this baseline.

**Comparative Model / Comparative Architecture**  
Alternative architectures compared against the baseline. In this project: ResNet50 pre-trained on RadImageNet (medical imaging weights, deeper network).

**RadImageNet**  
A large-scale dataset of medical images used for pre-training models on medical imaging tasks. Models pre-trained on RadImageNet may transfer better to MRI classification than ImageNet pre-trained models.

**ImageNet**  
A large-scale natural image dataset (1.2M images, 1000 classes) commonly used for pre-training computer vision models. Standard source of pre-trained weights, though not domain-specific for medical imaging.

---

## Project Structure Terms

**Role**  
A team member's primary area of responsibility. Six roles in this project: (1) Data Preprocessing, (2) Baseline Models, (3) Comparative Models, (4) Training Pipeline, (5) Evaluation, (6) Explainability.

**Interface**  
The contract between roles - the inputs a role expects and outputs it provides. Example: Role 1 (Data) provides a dataset with a `pos_weight` attribute that Role 4 (Training) consumes.

**Module**  
A Python file containing related functionality. Examples: `data_preprocessing_transformation.py`, `baseline_models.py`, `train.py`.

**Base Class**  
A parent class that encapsulates shared functionality. In this project, `MRNetBaseModel` contains common logic (max pooling, slice tracking, FC head, freeze/unfreeze utilities) used by both baseline and comparative models.

**Training Phase / Workflow Phase**  
Distinct stages in the model development pipeline:
- **Phase A:** All-slices training (baseline data)
- **Slice Extraction:** Model-guided identification of informative slices
- **Phase B:** Filtered training (reduced data)
- **Evaluation:** Performance comparison across all models
- **Triage:** Clinical interface development

Each phase has dependencies on previous phases and specific deliverables.

---

## Common Acronyms

- **ACL:** Anterior Cruciate Ligament
- **MRI:** Magnetic Resonance Imaging
- **CNN:** Convolutional Neural Network
- **FC:** Fully Connected (layer)
- **BCE:** Binary Cross-Entropy (loss function)
- **AUC:** Area Under Curve
- **ROC:** Receiver Operating Characteristic
- **PR:** Precision-Recall
- **TP/TN/FP/FN:** True Positive / True Negative / False Positive / False Negative
- **HPC:** High-Performance Computing
- **GPU:** Graphics Processing Unit
- **Top-K:** Top K elements (e.g., top-5 slices = 5 most important slices)

---

## Novel Methodology Terms (New in This Project)

These terms are specific to the novel contributions of this project and may not be found in standard MRNet literature:

**Model-Guided Slice Selection**  
A data curation approach where a trained model identifies which slices are most important for its predictions, and these slices are used to create a filtered dataset for retraining. Hypothesis: Training on only informative slices can improve performance by reducing noise and increasing focus on relevant anatomy.

**Dynamic Slice Selection / Per-Exam Selection**  
Selecting different slices for each exam based on that specific exam's features, rather than using fixed slice indices or positions across all exams. Each exam may have its tear at a different anatomical location, so the "most informative" slices vary per case.

**Slice Selection Independence**  
The principle that each condition-plane-architecture combination selects its own informative slices. ACL-sagittal identifies different slices than Meniscus-coronal because they look for different anatomical features. Similarly, ResNet18 and ResNet50 may attend to different patterns, so each selects independently.

**forward_with_slice_tracking()**  
A model method that extends the standard forward pass to return both the prediction (logit) and metadata about which slices contributed to max pooling. Technical implementation: stores slice indices during the max pooling operation and returns them alongside the prediction.

**Retraining on Filtered Data**  
The second training phase where the same model architecture is trained from scratch using only the top-K slices identified in the first phase. Different from fine-tuning (which continues training the same model) - this is a fresh training run with a modified dataset.

**Two-Mode Comparison**  
Evaluating each architecture in both training modes (all-slices and filtered) to isolate the effect of slice selection. Example: ResNet18-all-slices vs ResNet18-filtered tells us if slice selection helps. ResNet18-filtered vs ResNet50-filtered tells us if better architecture helps. This creates a 2×2 comparison matrix.

---

## Usage Notes

**When to use this document:**
- During cross-role communication and meetings
- When writing code comments and documentation
- When preparing reports and presentations
- When onboarding new team members

**Maintaining this document:**
- Update terminology as new concepts emerge during development
- Add clarifications if team members use terms inconsistently
- Keep definitions concise and accessible

**Not included in this document:**
- Implementation details (see code comments and module docstrings)
- Design decisions and trade-offs (see ADRs when created)
- Experimental results (see evaluation reports)
