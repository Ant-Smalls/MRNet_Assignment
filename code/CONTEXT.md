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

**End-to-End Fine-Tuning**  
Training the entire model (backbone + classification head) simultaneously from the start with a small learning rate (~1e-4). Allows pretrained features to adapt to the medical imaging domain while preserving useful learned patterns. Single-phase approach used in this project.

**pos_weight**  
A parameter of `BCEWithLogitsLoss` that upweights the positive class to handle class imbalance. Computed as `num_negative / num_positive`. For example, if ACL tears are 20% of the dataset (200 positive, 800 negative), `pos_weight = 800/200 = 4.0`.

**Checkpoint**  
A saved model state including weights and training metadata. In this project, we save only the **best checkpoint** (model with best validation AUC) for each training run, not intermediate or final checkpoints.

**Pilot Run**  
Training a single model first to estimate training time, memory requirements, and validate the training pipeline before submitting all jobs. Essential for batch job systems like SLURM.

**Pilot Run**  
Training a single model first to estimate training time, memory requirements, and validate the training pipeline before submitting all jobs. Essential for batch job systems like SLURM.

---

## Data & Preprocessing Terms

**R-CNN Semantic Cropping / Spatial Preprocessing**  
A one-time preprocessing step using Faster R-CNN to automatically detect and crop knee joint space from MRI volumes. Trained on manual annotations, then applied to entire dataset. Saves cropped volumes to a separate directory (`data/mrnet_cropped/`). Novel contribution that reduces noise and focuses models on anatomically relevant regions.

**Data Mode**  
The preprocessing variant of the dataset used for training or evaluation. Two modes:
- **Uncropped:** Original raw MRI data from `data/mrnet/`
- **Cropped:** R-CNN preprocessed data from `data/mrnet_cropped/`

**One-Time Preprocessing**  
Running a preprocessing pipeline (like R-CNN cropping) once before training begins and saving results to disk, rather than applying transformations during data loading. More efficient for expensive operations that don't change between runs.

**Augmentation**  
Random transformations applied to training images to increase dataset diversity and reduce overfitting. In this project:
- **Included:** Contrast/brightness jittering (simulates different MRI scanners), small rotations (±10°)
- **Excluded:** Horizontal/vertical flips (would incorrectly mirror anatomical sidedness), random crops (conflicts with R-CNN preprocessing)

**On-the-fly Augmentation**  
Augmentations applied during training when each batch is loaded, not pre-computed and saved to disk. Each epoch sees different random augmentations of the same images.

**Channel Stacking / 1→3 Channel Conversion**  
Repeating a single-channel grayscale image 3 times to create a 3-channel RGB-like tensor. Required to use ImageNet pre-trained models, which expect 3-channel inputs. Example: `(H, W)` → `(3, H, W)` by repeating along the channel dimension.

**Train/Val/Test Split**  
- **Train (80% of `train/` folder):** Used for model training (backpropagation, weight updates)
- **Validation (20% of `train/` folder):** Used for early stopping, hyperparameter tuning, and model selection
- **Test (provided `valid/` folder):** Held-out set for final evaluation only, never used during training or tuning

---

## Evaluation & Metrics Terms

**Prediction Fusion / Multi-Plane Fusion**  
Combining predictions from models trained on different planes (axial, coronal, sagittal) into a single prediction. In this project, using simple averaging: `final_prob = (prob_axial + prob_coronal + prob_sagittal) / 3`.

**Per-Plane Evaluation**  
Reporting metrics for individual plane models separately (e.g., ACL-axial: AUC 0.82, ACL-coronal: AUC 0.84, ACL-sagittal: AUC 0.86). Helps identify which planes are most informative for each condition.

**Fused Evaluation**  
Reporting metrics after combining predictions from all 3 planes (e.g., ACL-fused: AUC 0.87). This is the final model performance metric.

**2×2 Comparison Matrix**  
Experimental design structure that compares 4 training modes:
- Architecture dimension: ResNet18-ImageNet vs ResNet50-RadImageNet
- Data dimension: Uncropped vs Cropped
Enables analysis of main effects (architecture, preprocessing) and interaction effects (does cropping help one architecture more than the other?).

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

**Max-Pooled Slice / Most Informative Slice**  
The specific slice that contributed most to the final prediction after max pooling. Since max pooling selects maximum feature activations across slices, we can track which slice provided those maximums. This is the most "important" slice for Grad-CAM visualization and explainability. Identified at inference time, not used for training data filtering.

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
- **Training:** All 36 models trained in parallel (4 modes × 3 conditions × 3 planes)
- **Evaluation:** Performance comparison across 2×2 matrix
- **Triage:** Clinical interface development using best performing model

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

**R-CNN Preprocessing Pipeline / Semantic Joint Detection**  
A Faster R-CNN model trained on manually annotated middle slices to automatically detect and crop knee joint space from MRI volumes. Applied as one-time preprocessing before classification training. Novel contribution that focuses models on anatomically relevant regions and reduces background noise.

**Slice Tracking for Explainability**  
Using max pooling metadata to identify which slices contributed most to a model's prediction, enabling targeted visualization and interpretation. Unlike filtered training approaches, slice tracking is used purely for understanding model behavior, not for data curation.

**forward_with_slice_tracking()**  
A model method that extends the standard forward pass to return both the prediction (logit) and metadata about which slices contributed to max pooling. Technical implementation: stores slice indices during the max pooling operation and returns them alongside the prediction.

**2×2 Experimental Design**  
Systematic comparison of 4 training modes arranged in a matrix:
- Architecture: ResNet18-ImageNet vs ResNet50-RadImageNet
- Data: Uncropped vs Cropped
Enables measurement of main effects (architecture, preprocessing) and interaction effects (synergies between architectural and preprocessing choices).

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
