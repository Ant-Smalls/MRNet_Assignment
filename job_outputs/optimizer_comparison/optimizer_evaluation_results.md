# Optimizer Evaluation Results: AdamW vs. SGD

## Overview
This document summarizes the performance comparison between the **AdamW** and **SGD** optimizers across the MRNet dataset. The evaluation encompasses three pathologies (ACL, Meniscus, and Abnormality) using both the baseline architecture (AlexNet pre-trained on ImageNet) and the comparative architecture (DenseNet121 pre-trained on X-Ray Vision).

## Key Findings

1. **Optimizer Stability**: AdamW consistently outperformed SGD across all configurations. The adaptive learning rate of AdamW was critical for optimizing the deeper comparative models (DenseNet121). Under SGD, the comparative models suffered from probability calibration collapse, failing to converge effectively (e.g., yielding an AUC of 50.31% for ACL classification).
2. **Impact of Data Cropping**: The cropped dataset yielded superior performance compared to the uncropped dataset in 11 out of 12 experimental configurations, validating the hypothesis that isolating the region of interest improves model accuracy.

## Performance Metrics

| Optimizer | Pathology | Architecture | Uncropped AUC | Cropped AUC | Cropping Effect |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **AdamW** | **ACL** | AlexNet (Baseline) | 92.26% | 94.47% | +2.21% (Cropped Superior) |
| **AdamW** | **ACL** | DenseNet (Comparative) | 81.48% | 82.94% | +1.46% (Cropped Superior) |
| **AdamW** | **MENISCUS** | AlexNet (Baseline) | 80.60% | 82.21% | +1.61% (Cropped Superior) |
| **AdamW** | **MENISCUS** | DenseNet (Comparative) | 75.85% | 78.82% | +2.97% (Cropped Superior) |
| **AdamW** | **ABNORMAL** | AlexNet (Baseline) | 87.49% | 89.98% | +2.49% (Cropped Superior) |
| **AdamW** | **ABNORMAL** | DenseNet (Comparative) | 81.43% | 82.32% | +0.89% (Cropped Superior) |
| **SGD** | **ACL** | AlexNet (Baseline) | 86.11% | 87.77% | +1.66% (Cropped Superior) |
| **SGD** | **ACL** | DenseNet (Comparative) | 50.31% | 56.14% | +5.83% (Cropped Superior) |
| **SGD** | **MENISCUS** | AlexNet (Baseline) | 74.72% | 78.90% | +4.18% (Cropped Superior) |
| **SGD** | **MENISCUS** | DenseNet (Comparative) | 54.52% | 53.34% | -1.18% (Uncropped Superior) |
| **SGD** | **ABNORMAL** | AlexNet (Baseline) | 84.17% | 85.85% | +1.68% (Cropped Superior) |
| **SGD** | **ABNORMAL** | DenseNet (Comparative) | 63.16% | 68.67% | +5.51% (Cropped Superior) |

## Conclusion
Due to the poor convergence and lack of calibration observed in the SGD models, the full explainability pipeline (e.g., Grad-CAM heatmaps) was not executed for this optimizer. The final clinical triage reports and visualizations will exclusively feature the models trained with AdamW.
