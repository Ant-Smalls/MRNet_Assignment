import json
import base64
import pandas as pd
from pathlib import Path

curations = {
    "acl": [70, 77, 81, 4, 9, 17, 95, 116, 7, 76],
    "meniscus": [70, 84, 94, 3, 6, 65, 78, 97, 45, 13],
    "abnormal": [27, 29, 34, 0, 9, 11, 43, 73, 13, 14]
}

df = pd.read_csv("job_outputs/slice_inspection/all_patients_best_slices.csv")

def get_outcome(prob, lbl):
    pred = prob >= 0.5
    real = lbl == 1
    if pred and real: return "TP"
    if not pred and not real: return "TN"
    if pred and not real: return "FP"
    return "FN"

out_dir = Path("job_outputs/gradcam")
out_dir.mkdir(parents=True, exist_ok=True)

for data_mode in ["cropped", "uncropped"]:
    results = []
    
    for cond, pat_ids in curations.items():
        for plane in ["axial", "coronal", "sagittal"]:
            cases = []
            
            for pat in pat_ids:
                # Find the row in CSV (the CSV only has the 'best' plane for the condition)
                # But we need the pred_prob/slice_idx for THIS plane. 
                # Wait, the CSV ONLY contains best planes! 
                # If we need pred_prob for the non-best planes, it's not in the CSV!
                pass
