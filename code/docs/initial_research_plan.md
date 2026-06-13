# Initial Research Plan: MRNet Deep Learning Classification Pipeline

This document outlines the architectural decisions and pipeline strategies for the MRNet deep learning classification assignment. The goal of this project is to build a robust pipeline to classify knee MRI exams for ACL tears, meniscal tears, and general abnormalities.

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
**Decision:** Tailor augmentations to medical imaging constraints. Specifically, **remove `RandomHorizontalFlip`** and introduce contrast/brightness jittering.
**Rationale:**
- Unlike standard photographs, flipping knee MRIs horizontally mirrors medial/lateral anatomy, which could confuse the model (especially for specific meniscal tears).
- Contrast and brightness variations mimic the natural discrepancies found between different MRI machines (e.g., 1.5T vs 3.0T scanners).

## 6. Transfer Learning and Input Channels
**Decision:** Use **ImageNet pre-trained ResNet18** as the baseline, with **RadImageNet pre-trained ResNet50** as the comparative architecture.
**Rationale:**
- **Baseline (ResNet18 + ImageNet):** Standard, practical approach to leverage general vision weights. Input handling: stack 1-channel grayscale slices 3 times to mimic RGB.
- **Comparative (ResNet50 + RadImageNet):** Tests the hypothesis that medical imaging pre-training improves performance. RadImageNet models are trained on diverse radiological images and may better extract features from grayscale anatomical structures.
- **Architecture decision:** We focus on CNN-based architectures (ResNet variants) to maintain consistent max pooling slice tracking. Vision Transformers (e.g., DINOv2) are considered out of scope due to different attention-based slice selection mechanisms.

## 7. Model-Guided Slice Selection (Novel Methodology)
**Decision:** Implement a **two-phase training approach** that uses model feedback to identify and retain only the most informative slices.
**Rationale:**
- **Research Question:** Can model-guided slice selection improve both performance and interpretability compared to using all slices?
- **Clinical Motivation:** Not all slices in an MRI volume contain diagnostically relevant information. By identifying the most informative slices, we can potentially reduce noise, improve model focus, and enhance computational efficiency.

**Methodology:**
1. **Phase 1 - Baseline Training (All Slices):**
   - Train models on complete MRI volumes (all slices per exam)
   - Use `forward_with_slice_tracking()` to monitor which slices contribute to max pooling
   
2. **Phase 2 - Filtered Training (Top-K Slices):**
   - Analyze Phase 1 models to extract the **top-5 most informative slices per exam**
   - Slices are selected per-exam (dynamic) based on their contribution to max pooling activations
   - Retrain the same architecture on filtered datasets containing only selected slices
   
3. **Comparison:**
   - Compare performance: All-slices vs Filtered-slices for both baseline and comparative models
   - Expected outcomes: Performance improvement (reduced noise) OR similar performance (computational efficiency gain)

**Scope:**
- Slice selection is performed **independently** per condition-plane combination (9 selections for baseline, 9 for comparative)
- Each architecture (ResNet18 vs ResNet50-RadImageNet) selects its own informative slices
- **Top-K = 5 slices** selected per exam (from typical 20-40 slice volumes)

## 8. Comprehensive Evaluation and Interpretability
**Decision:** Evaluate models using clinical metrics, multi-plane fusion, and advanced interpretability tools.
**Rationale:**
- **Metrics:** Report AUC, Sensitivity (Recall), Specificity, and F1-Score with 95% confidence intervals to reflect clinical realities.
- **Curves:** Plot ROC and Precision-Recall (PR) curves for imbalanced class visualization.
- **Multi-Plane Fusion:** Combine predictions from all 3 planes (axial, coronal, sagittal) using simple averaging for final per-condition diagnosis.
- **Interpretability:** Implement **Grad-CAM** on the most informative slices identified by max pooling, confirming the model focuses on relevant joint anatomy.

## 9. Novel Clinical Application: Intelligent Triage Interface
**Decision:** Develop a **clinical triage visualization tool** that prioritizes cases for radiologist review based on model confidence and explainability.
**Rationale:**
- **Clinical Need:** Radiologists face large workloads. Prioritizing high-confidence predictions can improve workflow efficiency.
- **Interpretability Requirement:** Simply providing a confidence score is insufficient; clinicians need to see WHY the model made its prediction.

**Implementation:**
1. **Confidence-Based Ranking:**
   - Rank all test cases by fused prediction confidence (highest to lowest)
   - High-confidence positive predictions = urgent review priority
   
2. **Multi-Plane Visualization:**
   - For each case, display predictions from all 3 planes (axial, coronal, sagittal)
   - Show per-plane confidence scores and the fused confidence
   
3. **Slice-Level Explainability:**
   - Extract and display the **most informative slice** from each plane
   - Overlay Grad-CAM heatmaps to highlight anatomical regions driving the prediction
   
4. **Interface Format:**
   - Primary focus: Single condition (ACL tear detection) as proof of concept
   - Extensible to all 3 conditions (Meniscus, Abnormal) if time permits
   - Display format: Jupyter notebook or simple HTML interface with thumbnail grid

**Novel Contribution:** This system combines model confidence, slice selection, and explainability into a clinically-motivated workflow tool, going beyond traditional model evaluation.