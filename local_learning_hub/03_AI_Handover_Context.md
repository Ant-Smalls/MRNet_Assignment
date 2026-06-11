# AI Context Handover: MRNet Project

**Target:** Deep Learning Classification Pipeline for MRNet dataset (Knee MRIs).
**Goal:** 8-page MICCAI-style conference paper (25%) + Individual Viva/ePoster (15%).

## Current Architecture & Implementation
We have successfully implemented a professional-grade PyTorch pipeline covering Roles 4, 5, and 6 of the group project.

1. **Model:** `ResNet18` (Baseline Convolutional Neural Network).
2. **Training Strategy (`train.py`):** 
   - Two-phase training: Phase 1 freezes the ResNet backbone to train the classification head (10 epochs, lr=1e-3). Phase 2 unfreezes the entire network for fine-tuning (20 epochs, lr=1e-4).
   - Loss Function: `BCEWithLogitsLoss` using `pos_weight` to handle class imbalance (very few tear cases vs normal cases).
   - Optimizer: `Adam`.
   - Checkpointing: Automatically saves the `.pth` file for the epoch with the lowest Validation Loss.
3. **Data Augmentation (`data_preprocessing_transformation.py`):** Random affine transforms (rotation, translation) applied to training data to prevent overfitting.
4. **Evaluation (`evaluation.py`):** Currently calculates Loss and ROC-AUC. 
5. **Explainability (`explainability.py`):** Generates Grad-CAM heatmaps overlaying the model's spatial focus on the MRI slices to prove clinical validity and detect bias.

## Current Execution Status
- The user is currently running `MRNet_Colab_Training.ipynb` on a Google Colab T4 GPU overnight. 
- It is looping through all 9 baseline models (3 conditions × 3 planes).
- Checkpoints are saving directly to the user's mounted Google Drive.
- **Observations:** Models are successfully overfitting to the training set (Train Loss -> 0.000) and the checkpointing system is working perfectly to save the best generalized model (e.g., Coronal Validation Loss hit 0.54 at Epoch 20 after chaotic spikes).
