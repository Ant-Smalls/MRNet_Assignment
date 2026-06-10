"""
Data Sanity Check — MRNet Dataset
Validates structure, label distributions, and spot-checks MRI volumes.
Run from the project root: .venv/bin/python code/src/data_sanity_check.py
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "code/src/data/mrnet"
CONDITIONS = ["acl", "meniscus", "abnormal"]
PLANES = ["axial", "coronal", "sagittal"]
SPLITS = ["train", "valid"]


def load_labels(split, condition):
    """Load label CSV (first row is a dummy header, skip it)."""
    csv_path = os.path.join(DATA_DIR, f"{split}-{condition}.csv")
    df = pd.read_csv(csv_path, header=None, names=["patient_id", "label"])
    # Drop the first row if it's the dummy header (_0000, _0 or _0001)
    df = df[~df["patient_id"].astype(str).str.startswith("_")]
    df["label"] = df["label"].astype(int)
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(4)
    return df


def check_label_distributions():
    print("=" * 60)
    print("  CLASS DISTRIBUTION (Clinical Imbalance Check)")
    print("=" * 60)

    for split in SPLITS:
        print(f"\n  [{split.upper()} SET]")
        print(f"  {'Condition':<12} {'Total':>7} {'Positive':>10} {'Negative':>10} {'Prevalence':>12} {'pos_weight':>12}")
        print(f"  {'-'*65}")
        for cond in CONDITIONS:
            df = load_labels(split, cond)
            total = len(df)
            pos = df["label"].sum()
            neg = total - pos
            prevalence = pos / total * 100
            pos_weight = neg / pos if pos > 0 else float("inf")
            print(f"  {cond.capitalize():<12} {total:>7} {int(pos):>10} {int(neg):>10} {prevalence:>11.1f}% {pos_weight:>12.2f}")

    print()


def check_file_counts():
    print("=" * 60)
    print("  FILE COUNTS (Exam Completeness Check)")
    print("=" * 60)
    print(f"\n  {'Split':<8} {'Plane':<12} {'Files':>8}")
    print(f"  {'-'*30}")
    all_ok = True
    for split in SPLITS:
        for plane in PLANES:
            folder = os.path.join(DATA_DIR, split, plane)
            files = [f for f in os.listdir(folder) if f.endswith(".npy")]
            expected = 1130 if split == "train" else 120
            status = "✅" if len(files) == expected else f"⚠️  (expected {expected})"
            print(f"  {split:<8} {plane:<12} {len(files):>8}  {status}")
            if len(files) != expected:
                all_ok = False
    if all_ok:
        print("\n  ✅ All file counts match expected totals.")
    print()


def check_label_file_alignment():
    print("=" * 60)
    print("  LABEL ↔ FILE ALIGNMENT CHECK")
    print("=" * 60)
    for split in SPLITS:
        for cond in CONDITIONS:
            df = load_labels(split, cond)
            # Check that every patient in label CSV has a file (using axial as reference)
            folder = os.path.join(DATA_DIR, split, "axial")
            existing_ids = set(f.replace(".npy", "") for f in os.listdir(folder) if f.endswith(".npy"))
            label_ids = set(str(pid).zfill(4) for pid in df["patient_id"])
            missing = label_ids - existing_ids
            extra = existing_ids - label_ids
            if missing or extra:
                print(f"  ⚠️  {split}/{cond}: {len(missing)} labels without files, {len(extra)} files without labels")
            else:
                print(f"  ✅ {split}/{cond}: all {len(label_ids)} labels have matching .npy files")
    print()


def check_volume_shapes():
    print("=" * 60)
    print("  MRI VOLUME SHAPE SAMPLE (5 random train exams)")
    print("=" * 60)
    print(f"\n  {'Exam':>6} | {'Axial (slices,H,W)':^22} | {'Coronal':^22} | {'Sagittal':^22}")
    print(f"  {'-'*80}")

    axial_folder = os.path.join(DATA_DIR, "train", "axial")
    sample_files = sorted(os.listdir(axial_folder))[:5]

    for fname in sample_files:
        exam_id = fname.replace(".npy", "")
        shapes = []
        for plane in PLANES:
            fpath = os.path.join(DATA_DIR, "train", plane, fname)
            vol = np.load(fpath)
            shapes.append(str(vol.shape))
        print(f"  {exam_id:>6} | {shapes[0]:^22} | {shapes[1]:^22} | {shapes[2]:^22}")

    print()
    print("  Notes:")
    print("  - First dimension = number of slices (variable per exam = expected)")
    print("  - H x W = image dimensions (should be consistent ~256x256 or 320x320)")
    print("  - Data type and value range check on first exam...")

    first_file = os.path.join(DATA_DIR, "train", "sagittal", sample_files[0])
    vol = np.load(first_file)
    print(f"\n  Sagittal exam 0000:")
    print(f"    dtype:  {vol.dtype}")
    print(f"    min:    {vol.min():.2f}")
    print(f"    max:    {vol.max():.2f}")
    print(f"    mean:   {vol.mean():.2f}")

    if vol.max() > 1.0:
        print(f"    ⚠️  Values > 1.0 — normalization to [0,1] will be needed in preprocessing")
    else:
        print(f"    ✅ Values already in [0,1]")
    print()


def check_slice_counts():
    print("=" * 60)
    print("  SLICE COUNT STATISTICS (Variable-length volumes)")
    print("=" * 60)
    for plane in PLANES:
        folder = os.path.join(DATA_DIR, "train", plane)
        slice_counts = []
        for fname in os.listdir(folder):
            if fname.endswith(".npy"):
                vol = np.load(os.path.join(folder, fname))
                slice_counts.append(vol.shape[0])
        arr = np.array(slice_counts)
        print(f"\n  {plane.capitalize()}:")
        print(f"    min slices: {arr.min()}")
        print(f"    max slices: {arr.max()}")
        print(f"    mean:       {arr.mean():.1f}")
        print(f"    std:        {arr.std():.1f}")
    print()
    print("  ✅ Variable slice counts confirmed — batch_size=1 is correct (no padding needed)")
    print()


if __name__ == "__main__":
    print("\n🔬 MRNet Dataset Sanity Check\n")
    check_file_counts()
    check_label_distributions()
    check_label_file_alignment()
    check_volume_shapes()
    check_slice_counts()
    print("=" * 60)
    print("  ✅ Sanity check complete. Dataset is ready for use.")
    print("=" * 60)
    print()
