# MRNet Dataset Auto-Cropper Instructions

This guide explains how to use the automated Faster R-CNN joint detection pipeline to crop the raw MRNet 3D volumes.

## Required Files
To run the cropping pipeline, you only need:
1. **`code/src/crop_mrnet_dataset.py`** (Already in this branch)
2. **`joint_detector.pth`** (Model weights - shared directly with you since the file is >150MB and too large for GitHub)

---

## Setup & Environment
The script requires `torch`, `torchvision`, `numpy`, and `tqdm`. Ensure these are installed in your Python environment:
```bash
pip install torch torchvision numpy tqdm
```

---

## How to Run the Cropper

By default, the script expects:
- The model weights to be at `models/joint_detector.pth`
- The raw MRNet `.npy` files to be at `data/mrnet/` (e.g., structured with subdirectories like `train/axial/0000.npy`, etc.)
- The cropped output will be saved to `data/mrnet_cropped/` (maintaining the original subdirectory structure)

Run the script using:
```bash
python code/src/crop_mrnet_dataset.py
```

### Customizing Paths
You can override the default paths using command-line arguments:
```bash
python code/src/crop_mrnet_dataset.py \
    --model_path /path/to/joint_detector.pth \
    --input_dir /path/to/raw/mrnet/ \
    --output_dir /path/to/save/cropped_mrnet/
```

---

## How the Algorithm Works
1. For each 3D volume, the script extracts the **middle slice** (which reliably shows the knee joint space).
2. It runs the Faster R-CNN model on the middle slice to locate the bounding box of the joint.
3. It applies a **15px safety padding** around the detected joint area.
4. It crops the entire 3D volume (all slices) to those exact bounding box coordinates to ensure consistent spatial alignment across the sequence.
5. The cropped volume is saved to the output directory as a `.npy` file.
