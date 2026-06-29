# MRNet: Deep Learning for Knee MRI Abnormality Detection

A comprehensive deep learning system for automated detection of knee pathologies from MRI scans, featuring preprocessing methods, multi-architecture comparisons, and clinical explainability tools.

## Overview

This project implements and extends the MRNet architecture for automated detection of three knee conditions from MRI volumes:
- **ACL (Anterior Cruciate Ligament) tears**
- **Meniscal tears** 
- **General abnormalities**

### Key Features

- **Multi-Architecture Comparison**: Baseline AlexNet (ImageNet pretraining) vs. DenseNet121 (chest X-ray pretraining via torchxrayvision)
- **Novel R-CNN Preprocessing**: Automated knee joint space detection and cropping using Faster R-CNN
- **Multi-Plane Fusion**: Combines predictions from axial, coronal, and sagittal views
- **Explainability**: Grad-CAM visualizations on most informative slices
- **Clinical Triage Dashboard**: Interactive HTML interface for radiologist review prioritization
- **2×2 Experimental Design**: Systematic comparison of architectures × preprocessing methods

## Project Structure

```
MRNet_assignment/
├── code/
│   ├── src/
│   │   ├── train.py                    # Training pipeline with SLURM support
│   │   ├── evaluation.py               # Multi-plane evaluation & bootstrapped metrics
│   │   ├── explainability.py           # Grad-CAM heatmap generation
│   │   ├── modules/
│   │   │   ├── baseline_models.py      # AlexNet-based architecture
│   │   │   ├── comparative_models.py   # DenseNet121 with X-ray pretraining
│   │   │   ├── data_preprocessing_transformation.py
│   │   │   └── R-CNN_cropping_model/   # Knee joint detection system
│   │   ├── reporting/
│   │   │   └── generate_triage_dashboard.py  # Clinical interface
│   │   └── data/
│   │       ├── mrnet/                  # Original MRI volumes
│   │       └── mrnet_cropped/          # R-CNN preprocessed data
│   ├── docs/                           # Project documentation
│   └── CONTEXT.md                      # Terminology & domain concepts
├── sonic_scripts/                      # SLURM job submission scripts
├── job_outputs/                        # Evaluation results & reports
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt
```

### Dependencies

Core packages:
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Vision models and transforms
- `torchxrayvision>=1.0.0` - DenseNet121 with chest X-ray pretraining
- `grad-cam>=1.4.0` - Explainability
- `scikit-learn>=1.3.0` - Metrics
- `pandas>=2.0.0`, `numpy>=1.24.0` - Data processing
- `matplotlib>=3.7.0` - Visualization

### Training a Model

#### Single Model
```bash
python3 code/src/train.py \
    --condition acl \
    --plane sagittal \
    --architecture baseline \
    --data_mode cropped \
    --epochs 30 \
    --lr 1e-4
```

#### SLURM Batch Training
```bash
# Train all comparative models (18 jobs: 3 conditions × 3 planes × 2 data modes)
sbatch sonic_scripts/submit_all_comparative.sh
```

### Evaluation

```bash
# Evaluate single condition with multi-plane fusion
python3 code/src/evaluation.py \
    --condition acl \
    --architectures baseline comparative \
    --data_modes uncropped cropped \
    --n_bootstraps 1000

# Generate master evaluation table (all conditions)
python3 code/src/evaluation.py \
    --condition all \
    --architectures baseline comparative \
    --data_modes uncropped cropped
```

### Explainability

```bash
# Generate Grad-CAM visualizations
python3 code/src/explainability.py \
    --checkpoint_dir all_checkpoints/ \
    --data_dir code/src/data/ \
    --output_dir job_outputs/gradcam/ \
    --data_mode cropped \
    --n_cases 10
```

### Clinical Triage Dashboard

```bash
# Generate interactive HTML dashboard
python3 code/src/reporting/generate_triage_dashboard.py \
    --results_dir job_outputs/results/ \
    --gradcam_dir job_outputs/gradcam/ \
    --output job_outputs/triage_dashboard.html
```

## Model Architecture

### MRNet Base Architecture

```
Input: (B, S, 3, H, W)  # Batch × Slices × RGB × Height × Width
    ↓
Backbone CNN (per-slice feature extraction)
    ↓
Global Average Pooling → (B, S, 256)
    ↓
Max Pooling across slices → (B, 256)
    ↓
Fully Connected (256 → 1)
    ↓
Output: (B, 1)  # Binary classification logit
```

### Slice Tracking for Explainability

During max pooling, the model tracks which slice contributed each feature dimension. This enables:
- Identification of the most informative slice per exam
- Targeted Grad-CAM visualization on diagnostically relevant slices
- Clinical interpretability through anatomically grounded explanations

## Experimental Design: 2×2 Comparison Matrix

| Architecture | Uncropped | Cropped |
|-------------|-----------|---------|
| **Baseline** (AlexNet-ImageNet) | Mode 1 | Mode 2 |
| **Comparative** (DenseNet121-XRayVision) | Mode 3 | Mode 4 |

This design enables analysis of:
- **Main effect (Architecture)**: Does deeper network + medical pretraining help?
- **Main effect (Preprocessing)**: Does R-CNN cropping improve performance?
- **Interaction effect**: Do certain architectures benefit more from cropping?

## Evaluation Metrics

### Per-Plane Metrics
- AUC-ROC with 95% bootstrap confidence intervals
- Sensitivity, Specificity, F1-Score
- Precision-Recall curves

### Multi-Plane Fusion
- Simple averaging: `p_fused = (p_axial + p_coronal + p_sagittal) / 3`
- Aggregated metrics across all three planes

### Statistical Testing
- Bootstrap resampling (1000 iterations) for confidence intervals
- Wilcoxon signed-rank test for pairwise model comparisons

## Contributions

### 1. R-CNN Semantic Cropping
- Faster R-CNN trained on manually annotated knee joint regions
- One-time preprocessing: detects and crops anatomically relevant areas
- Reduces background noise and focuses models on diagnostic regions

### 2. Medical Domain Pretraining
- DenseNet121 pretrained on chest X-rays (via `torchxrayvision`)
- Hypothesized better transfer to MRI than natural images
- Evaluated against standard ImageNet pretraining

### 3. Slice Tracking Mechanism
- Extended forward pass: `forward_with_slice_tracking()`
- Returns prediction + metadata about contributing slices
- Enables explainability without modifying training procedure

### 4. Clinical Triage Interface
- Confidence-based case ranking for radiologist review
- Multi-plane visualization with synchronized Grad-CAM overlays
- True/False Positive/Negative filtering
- Fully self-contained HTML (works offline)

## Results

*(Results are stored in `job_outputs/` after running evaluation scripts)*

### Key Findings
1. **Cropping improves all models**: Preprocessing shows consistent benefits
2. **Sagittal plane most informative for ACL**: Aligns with clinical knowledge

## Documentation

- **[CONTEXT.md](code/CONTEXT.md)**: Domain terminology and glossary
- **[code/docs/](code/docs/)**: Technical documentation
  - Project roles & responsibilities
  - Initial research plan

## Hyperparameters

### Default Training Configuration
```python
LEARNING_RATE = 1e-4      # AdamW optimizer
WEIGHT_DECAY = 1e-4       # L2 regularization
BATCH_SIZE = 1            # Due to variable slice counts
EPOCHS = 50               # Single-phase training
PATIENCE = 10             # Early stopping
```

### Data Augmentation
- Contrast/brightness jitter (simulate scanner variations)
- Small rotations (±10°)
- **Not used**: Flips (preserve anatomical laterality), random crops (conflicts with R-CNN preprocessing)

## Clinical Workflow Integration

The triage dashboard simulates a real-world radiologist workflow:

1. **Priority Ranking**: Cases sorted by model confidence (highest first)
2. **Explainability**: Grad-CAM highlights regions of interest
3. **Confidence Calibration**: Threshold tuning for sensitivity/specificity trade-offs
4. **Case Filtering**: Focus on TP/FP/TN/FN for model validation

## Citation

Relevant citations for this work include:

### Original MRNet Paper
```bibtex
@article{bien2018deep,
  title={Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet},
  author={Bien, N. and Rajpurkar, P. and Ball, R.L. and Irvin, J. and Park, A. and Jones, E. and Bereket, M. and Patel, B.N. and Yeom, K.W. and Shpanskaya, K. and Halabi, S. and Zucker, E. and Fanton, G. and Amanatullah, D.F. and Beaulieu, C.F. and Riley, G.M. and Stewart, R.J. and Blankenberg, F.G. and Larson, D.B. and Jones, R.H. and Langlotz, C.P. and Ng, A.Y. and Lungren, M.P.},
  journal={PLOS Medicine},
  volume={15},
  number={11},
  pages={e1002699},
  year={2018}
}
```

### Transfer Learning & Domain Adaptation
```bibtex
@inproceedings{yosinski2014,
  title={How transferable are features in deep neural networks?},
  author={Yosinski, J. and Clune, J. and Bengio, Y. and Lipson, H.},
  booktitle={Advances in Neural Information Processing Systems},
  volume={27},
  year={2014}
}

@article{davila2024,
  title={Comparison of fine-tuning strategies for transfer learning in medical image classification},
  author={Davila, A. and Colan, J. and Hasegawa, Y.},
  journal={arXiv preprint arXiv:2406.10050},
  year={2024}
}
```

### Model Robustness & Shortcut Learning
```bibtex
@article{geirhos2020shortcut,
  title={Shortcut learning in deep neural networks},
  author={Geirhos, R. and Jacobsen, J.H. and Michaelis, C. and Zemel, R. and Brendel, W. and Bethge, M. and Wichmann, F.A.},
  journal={Nature Machine Intelligence},
  volume={2},
  number={11},
  pages={665--673},
  year={2020}
}

@article{zech2018variable,
  title={Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study},
  author={Zech, J.R. and Badgeley, M.A. and Liu, M. and Costa, A.B. and Titano, J.J. and Oermann, E.K.},
  journal={PLOS Medicine},
  volume={15},
  number={11},
  pages={e1002683},
  year={2018}
}

@article{degrave2021ai,
  title={AI for radiographic COVID-19 detection selects shortcuts over signal},
  author={DeGrave, A.J. and Janizek, J.D. and Lee, S.I.},
  journal={Nature Machine Intelligence},
  volume={3},
  pages={610--619},
  year={2021}
}
```

### Knee MRI & Clinical Analysis
```bibtex
@article{yalcin2025moon,
  title={Meniscus tears associated with ACL injuries: The Multicenter Orthopaedic Outcomes Network (MOON) experience},
  author={Yalcin, S. and Sheean, A.J. and Jones, M.H. and Spindler, K.P.},
  journal={Operative Techniques in Orthopaedics},
  volume={35},
  pages={101188},
  year={2025}
}

@article{chen2026radiomic,
  title={Interpretability and individuality in knee MRI: Patient-specific radiomic fingerprint with reconstructed healthy personas},
  author={Chen, Y. and Ni, S. and Li, S. and Saeed, S.U. and Ivanova, A. and Hargunani, R. and Huang, J. and Liu, C. and Hu, Y.},
  journal={arXiv preprint arXiv:2601.08604},
  year={2026}
}

@article{systematicreview2022,
  title={Knee injury detection using deep learning on MRI studies: A systematic review},
  author={Siouras, A. and Moustakidis, S. and Giannakidis, A. and Chalatsis, G. and Liampas, I. and Vlychou, M. and Hantes, M. and Tasoulis, S. and Tsaopoulos, D.},
  journal={Diagnostics},
  volume={12},
  number={2},
  pages={537},
  year={2022}
}
```

### Explainability & Interpretability
```bibtex
@inproceedings{selvaraju2017gradcam,
  title={Grad-CAM: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, R.R. and Cogswell, M. and Das, A. and Vedantam, R. and Parikh, D. and Batra, D.},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages={618--626},
  year={2017}
}

@inproceedings{adebayo2018sanity,
  title={Sanity checks for saliency maps},
  author={Adebayo, J. and Gilmer, J. and Muelly, M. and Goodfellow, I. and Hardt, M. and Kim, B.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={9525--9536},
  year={2018}
}
```

### Object Detection & Preprocessing
```bibtex
@inproceedings{girshick2014,
  title={Rich feature hierarchies for accurate object detection and semantic segmentation},
  author={Girshick, R. and Donahue, J. and Darrell, T. and Malik, J.},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={580--587},
  year={2014}
}

@inproceedings{ren2015faster,
  title={Faster R-CNN: Towards real-time object detection with region proposal networks},
  author={Ren, S. and He, K. and Girshick, R. and Sun, J.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={91--99},
  year={2015}
}
```

### Deep Learning Foundations
```bibtex
@inproceedings{krizhevsky2012imagenet,
  title={ImageNet classification with deep convolutional neural networks},
  author={Krizhevsky, A. and Sutskever, I. and Hinton, G.E.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={1097--1105},
  year={2012}
}
```

### Optimization & Statistical Methods
```bibtex
@inproceedings{loshchilov2019,
  title={Decoupled weight decay regularization},
  author={Loshchilov, I. and Hutter, F.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}

@article{youden1950,
  title={Index for rating diagnostic tests},
  author={Youden, W.J.},
  journal={Cancer},
  volume={3},
  number={1},
  pages={32--35},
  year={1950}
}

@book{efron1993,
  title={An Introduction to the Bootstrap},
  author={Efron, B. and Tibshirani, R.J.},
  publisher={Chapman \& Hall},
  address={New York},
  year={1993}
}
```

## License

This project is for academic/educational purposes as part of the UCD MSc AI in Medicine and Medical Research, AI in Medical Imaging Module.

## Additional Acknowledgments

- **Stanford ML Group** for the original MRNet architecture and dataset
- **torchxrayvision** team for medical imaging pretraining weights
- **UCD School** for computational resources (Sonic HPC cluster)
- **Grad-CAM authors** for explainability implementation

---

**Last Updated**: June 2026  
