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
**Decision:** Use **Max Pooling** across the slice dimension as the baseline. 
**Rationale:** 
- Max pooling is a robust baseline that answers the question: "Is the strongest feature of a tear present in ANY of these slices?"
- *Iterative Improvement:* We will explore more advanced sequence modeling like Attention mechanisms or Bidirectional LSTMs, which could better characterize disease patterns that span across multiple slices.

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
**Decision:** Use **ImageNet pre-trained ResNet18** as the baseline, handled by stacking the 1-channel grayscale MRI slices 3 times to mimic RGB inputs.
**Rationale:**
- This is the standard, practical approach to leverage ImageNet weights without modifying the network architecture.
- *Iterative Improvement:* We will compare this ImageNet baseline against domain-specific models, such as **RadImageNet**, which are pre-trained natively on medical scans and may yield better feature extraction for grayscale anatomical structures.

## 7. Comprehensive Evaluation and Interpretability
**Decision:** Evaluate the model using a broad suite of clinical metrics and interpretability tools.
**Rationale:**
- **Metrics:** Report AUC, Sensitivity (Recall), Specificity, and F1-Score to reflect clinical realities (e.g., the high cost of false negatives).
- **Curves:** Plot ROC and Precision-Recall (PR) curves, as PR curves are highly informative for imbalanced classes like ACL.
- **Interpretability:** Implement **Grad-CAM** on the max-pooled slice. Since we use max pooling, we can isolate the single slice that drove the prediction and generate a heatmap to confirm the model is focusing on relevant joint anatomy rather than background artifacts.