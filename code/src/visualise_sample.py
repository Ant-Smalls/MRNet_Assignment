"""
Quick visualiser — shows one sagittal exam with its label.
Run: .venv/bin/python code/src/visualise_sample.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, save to file
import matplotlib.pyplot as plt
import pandas as pd, os

DATA_DIR = "code/src/data/mrnet"

def load_labels(split, condition):
    csv_path = os.path.join(DATA_DIR, f"{split}-{condition}.csv")
    df = pd.read_csv(csv_path, header=None, names=["patient_id", "label"])
    df = df[~df["patient_id"].astype(str).str.startswith("_")]
    df["label"] = df["label"].astype(int)
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(4)
    return df

acl_labels = load_labels("train", "acl")
# Pick one positive (ACL tear) and one negative
pos_id = acl_labels[acl_labels["label"]==1]["patient_id"].iloc[0]
neg_id = acl_labels[acl_labels["label"]==0]["patient_id"].iloc[0]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.patch.set_facecolor("#0d0d0d")
fig.suptitle("MRNet Dataset — Sample MRI Slices\n(Sagittal plane, middle slice)", 
             color="white", fontsize=14, fontweight="bold", y=0.98)

for row, (exam_id, label) in enumerate([(pos_id, "ACL TEAR (+ve)"), (neg_id, "Normal (-ve)")]):
    colour = "#ff6b6b" if row == 0 else "#6bffb8"
    for col, plane in enumerate(["axial", "coronal", "sagittal"]):
        vol = np.load(os.path.join(DATA_DIR, "train", plane, f"{exam_id}.npy"))
        mid = vol.shape[0] // 2
        ax = axes[row][col]
        ax.imshow(vol[mid], cmap="gray", vmin=0, vmax=255)
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2)
        title = f"{plane.capitalize()} — slice {mid}/{vol.shape[0]-1}"
        if col == 1:
            title = f"Exam {exam_id}  |  {label}\n{title}"
        ax.set_title(title, color=colour if col==1 else "white", fontsize=8.5, pad=6)

plt.tight_layout(rect=[0,0,1,0.96])
out_path = "code/src/sample_mri.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
