# Initial Research Plan: MRNet Deep Learning Classification Pipeline

This document outlines the architectural decisions and pipeline strategies for the MRNet deep learning classification assignment. The goal is to build a robust pipeline to classify knee MRI exams for ACL tears, meniscal tears, and general abnormalities.

## Research Questions

1. **Does R-CNN semantic joint cropping improve classification performance?**
   - Compare models trained on uncropped vs cropped data
   
2. **Does medical imaging pretraining improve performance compared to natural image pretraining?**
   - Compare ResNet18-ImageNet vs ResNet50-RadImageNet

## 0. R-CNN Semantic Joint Preprocessing (Novel Methodology)

**Decision:** Use Faster R-CNN to automatically crop MRI volumes to focus on knee joint space before classification training.

**Rationale:**
- Mimics radiologist attention - focus on anatomically relevant region
- Reduces noise from surrounding tissue and image artifacts
- Improves spatial consistency across different patient scans
- Trained on manual annotations, applied to entire dataset

**Implementation:**
1. **Training:** Faster R-CNN trained on manually annotated middle slices
2. **Inference:** For each 3D volume, detect joint on middle slice
3. **Cropping:** Apply bounding box (with 15px padding) to all slices in volume
4. **One-time preprocessing:** Save cropped volumes to `data/mrnet_cropped/`

**Experimental Design:** Compare classification performance on original vs cropped data to measure preprocessing impact.

## 1. Task Architecture: Independent Binary Models
**Decision:** Train **separate models** for each classification task (ACL, Meniscus, Abnormal) rather than a single multi-label model.
**Rationale:** 
- Matches the methodology of the original MRNet paper.
- The tutorial notebook provides a baseline designed for isolated binary classification.
- Simplifies handling of unique class imbalances specific to each condition.

## 2. Spatial Dimension Strategy: Independent Plane Models
**Decision:** Train **separate models** for each imaging plane (Axial, Coronal, Sagittal) and combine their predictions at the end, resulting in 9 models total (3 conditions × 3 planes).
**Rationale:** 
- A unified multi-view model that takes all three planes simultaneously is complicated by variable slice counts across exams and spatial misalignment.
- Passing three 3D volumes into a single model simultaneously runs a high risk of out-of-memory (OOM) errors.
- Combining features or predictions later allows each model to specialize in the features most visible from its respective angle.

## 3. Slice Aggregation Strategy
**Decision:** Use **Max Pooling** across the slice dimension as the baseline and primary aggregation method.
**Rationale:** 
- Max pooling is a robust baseline that answers the question: "Is the strongest feature of a tear present in ANY of these slices?"
- Max pooling provides natural interpretability by tracking which slices contributed most to the prediction.
- We will maintain max pooling throughout all experiments to ensure fair comparisons between baseline and comparative models.
- *Note:* More advanced aggregation methods (Attention mechanisms, Bidirectional LSTMs, or multi-scale pooling) are considered out of scope to maintain focus on slice selection and clinical application novelty.

## 4. Class Imbalance Handling
**Decision:** Use a **Weighted Loss Function** (`BCEWithLogitsLoss` with weights proportional to the negative/positive ratio) as the baseline.
**Rationale:**
- The dataset is heavily imbalanced (e.g., ACL tears are only 23.3% of the dataset).
- Weighting penalizes the model appropriately for missing the minority class without dropping data (undersampling) or increasing training time unnecessarily (oversampling).
- *Iterative Improvement:* We may experiment with Focal Loss to focus the model on the "hardest" examples.

## 5. Medical Image Data Augmentation
**Decision:** Tailor augmentations to medical imaging constraints. Specifically, **remove horizontal/vertical flips** and use contrast/brightness jittering and small rotations.

**Rationale:**
- Flipping knee MRIs mirrors medial/lateral or superior/inferior anatomy, which could confuse the model
- Contrast and brightness variations mimic natural discrepancies between MRI machines (e.g., 1.5T vs 3.0T scanners)
- Small rotations (±10°) add variability while preserving anatomical orientation
- No random crops during training (would conflict with R-CNN preprocessing)

**Augmentations Applied:**
- ✅ Contrast/brightness jittering
- ✅ Random rotations (±10°)
- ❌ No horizontal flips
- ❌ No vertical flips
- ❌ No random crops (keep commented in code for potential future use)

## 6. Transfer Learning and Training Strategy
**Decision:** Use **ImageNet pre-trained ResNet18** as the baseline, with **RadImageNet pre-trained ResNet50** as the comparative architecture. Train with single-phase end-to-end fine-tuning.

**Rationale:**
- **Baseline (ResNet18 + ImageNet):** Standard approach to leverage general vision weights. Input handling: stack 1-channel grayscale slices 3 times to mimic RGB.
- **Comparative (ResNet50 + RadImageNet):** Tests hypothesis that medical imaging pre-training improves performance. RadImageNet models are trained on diverse radiological images and may better extract features from grayscale anatomical structures.
- **Single-phase training:** End-to-end fine-tuning with small learning rate (~1e-4) adapts pretrained features to medical imaging domain without complexity of multi-phase training.
- **Architecture focus:** CNN-based architectures (ResNet variants) for consistency with max pooling slice aggregation. Vision Transformers considered out of scope.

## 7. Slice Tracking for Explainability
**Decision:** Implement slice tracking mechanism to identify which slices contribute most to model predictions, used for explainability and visualization (not for filtered training).

**Rationale:**
- **Research Purpose:** Understand which slices the model relies on for predictions
- **Clinical Value:** Provide interpretable explanations by highlighting most informative slices
- **Triage Tool:** Enable slice-level visualization in clinical interface

**Methodology:**
1. **Slice Tracking Implementation:**
   - Add `forward_with_slice_tracking()` method to model
   - During max pooling, track which slice contributed maximum activation for each feature
   - Returns both prediction and slice indices: `(logits, slice_indices)`

2. **Usage for Explainability (Role 6):**
   - At inference time, identify most informative slice per exam
   - Generate Grad-CAM heatmap on that specific slice
   - Display in triage interface with confidence scores

**Note:** Unlike original plan, we do NOT retrain models on filtered top-K slices. Slice tracking is purely for understanding and explainability.

## 8. Comprehensive Evaluation and Comparison
**Decision:** Evaluate models using clinical metrics, multi-plane fusion, and structured 2×2 comparison matrix.

**Rationale:**
- **Metrics:** Report AUC, Sensitivity (Recall), Specificity, and F1-Score with 95% confidence intervals to reflect clinical realities.
- **Curves:** Plot ROC and Precision-Recall (PR) curves for imbalanced class visualization.
- **Multi-Plane Fusion:** Combine predictions from all 3 planes (axial, coronal, sagittal) using simple averaging for final per-condition diagnosis.
- **Structured Comparison:** 2×2 matrix design enables isolation of main effects and interaction effects.

**Experimental Design:**

| Architecture | Uncropped Data | Cropped Data |
|--------------|----------------|--------------|
| ResNet18-ImageNet | Baseline | Crop Effect |
| ResNet50-RadImageNet | Pretrain Effect | Best (hypothesis) |

**Comparisons:**
1. **Cropping Effect:** Uncropped vs Cropped (within each architecture)
2. **Pretraining Effect:** ImageNet vs RadImageNet (within each data mode)
3. **Interaction Effects:** Does cropping help more for one architecture? Does pretraining help more for one data mode?

**Total Models:** 36 (4 modes × 3 conditions × 3 planes)

## 9. Clinical Triage Interface (Novel Application)
**Decision:** Develop a **clinical triage visualization tool** that prioritizes cases for radiologist review based on model confidence and explainability.

**Rationale:**
- **Clinical Need:** Radiologists face large workloads. Prioritizing high-confidence predictions can improve workflow efficiency.
- **Interpretability Requirement:** Clinicians need to see WHY the model made its prediction, not just a confidence score.

**Implementation:**
1. **Model Selection:**
   - Use best performing model from 2×2 comparison (hypothesis: RadImageNet + cropped)
   - Let evaluation results determine actual best model

2. **Confidence-Based Ranking:**
   - Rank all test cases by fused prediction confidence (highest to lowest)
   - High-confidence positive predictions = urgent review priority

3. **Multi-Plane Visualization:**
   - For each case, display predictions from all 3 planes (axial, coronal, sagittal)
   - Show per-plane confidence scores and the fused confidence

4. **Slice-Level Explainability:**
   - Use slice tracking to identify most informative slice from each plane
   - Overlay Grad-CAM heatmaps to highlight anatomical regions driving the prediction

5. **Interface Format:**
   - Jupyter notebook with interactive widgets OR simple HTML interface
   - Display top-N cases (N=20-50)

6. **Scope:**
   - Primary focus: ACL tear detection (all 3 planes) as proof of concept
   - Extensible to all 3 conditions (Meniscus, Abnormal) if time permits

**Novel Contribution:** This system combines model confidence, slice tracking, and explainability into a clinically-motivated workflow tool, going beyond traditional model evaluation.