# Future Roadmap: MRNet Pipeline

Based on the feedback from NotebookLM and the Professor's grading rubric, here is the exact roadmap for the next coding session:

### 1. Update Evaluation Metrics (High Priority)
The professor explicitly warned against relying solely on ROC-AUC due to class imbalance.
- **Task:** Modify `evaluation.py` to calculate and output:
  - Precision
  - Recall
  - F1-Score
  - Precision-Recall AUC (PR-AUC)

### 2. Implement Comparative Models (Role 3)
The assignment guidelines mandate implementing "different architectures" to compare against the baseline.
- **Task:** Create `modules/comparative_models.py`.
- **Task:** Add `DenseNet121` (or another SOTA medical imaging model).
- **Task:** Run a training comparison between ResNet18 and DenseNet121 and analyze why one outperforms the other.

### 3. Hyperparameter Tuning & Stability
We observed chaotic training behavior (Val Loss spiking from 0.54 to 6.06 and back to 0.54 in Phase 2).
- **Task:** Explore adjusting the Learning Rate, or implement a Learning Rate Scheduler (e.g., `ReduceLROnPlateau`) to stabilize the fine-tuning phase.
- **Task:** Test alternative optimizers (like `SGD` with momentum) as suggested by the assignment rubric.

### 4. Paper & ePoster Generation
Once the models finish training overnight:
- **Task:** Download the `.pth` checkpoints from Google Drive to the local Mac.
- **Task:** Run `explainability.py` locally to generate the Grad-CAM heatmaps for the ePoster.
- **Task:** Run `evaluation.py` to generate the final tables for the MICCAI paper.
