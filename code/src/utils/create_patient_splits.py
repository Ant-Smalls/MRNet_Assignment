import json
import random
import pandas as pd
from pathlib import Path

def create_patient_splits(csv_path, output_path, train_ratio=0.8, seed=42):
    """Generate patient-level train/val splits (80/20). Patient IDs stored as integers; 
    convert to zero-padded filenames (e.g., 65 -> '0065.npy') when loading data."""
    df = pd.read_csv(csv_path, header=0)
    patient_ids = df.iloc[:, 0].unique().tolist()
    
    random.seed(seed)
    random.shuffle(patient_ids)
    
    split_idx = int(len(patient_ids) * train_ratio)
    train_patients = sorted(patient_ids[:split_idx])
    val_patients = sorted(patient_ids[split_idx:])
    
    splits = {
        'train': train_patients,
        'val': val_patients
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Patient splits created:")
    print(f"  Total patients: {len(patient_ids)}")
    print(f"  Train: {len(train_patients)} ({len(train_patients)/len(patient_ids)*100:.1f}%)")
    print(f"  Val: {len(val_patients)} ({len(val_patients)/len(patient_ids)*100:.1f}%)")
    print(f"  Saved to: {output_path}")

def verify_patient_files(splits_path, data_root):
    """Verify that each patient has axial, coronal, and sagittal .npy files. 
    Both train/val splits use files from mrnet/train/ directory."""
    with open(splits_path) as f:
        splits = json.load(f)
    
    data_root = Path(data_root)
    planes = ['axial', 'coronal', 'sagittal']
    missing_files = []
    
    for split_name in ['train', 'val']:
        for patient_id in splits[split_name]:
            filename = f"{patient_id:04d}.npy"
            
            for plane in planes:
                file_path = data_root / 'mrnet' / 'train' / plane / filename
                
                if not file_path.exists():
                    missing_files.append({
                        'split': split_name,
                        'patient_id': patient_id,
                        'plane': plane,
                        'expected_path': str(file_path)
                    })
    
    if missing_files:
        print(f"\n Found {len(missing_files)} missing files:")
        for missing in missing_files[:10]:
            print(f"  - {missing['split']}/{missing['plane']}: patient {missing['patient_id']}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        return False
    else:
        total_checked = sum(len(splits[s]) for s in ['train', 'val']) * 3
        print(f"\n All {total_checked} files verified (3 planes × {sum(len(splits[s]) for s in ['train', 'val'])} patients)")
        return True

if __name__ == '__main__':
    csv_path = '../data/mrnet/train-acl.csv'
    output_path = '../data/patient_splits_training.json'
    data_root = '../data/'
    
    create_patient_splits(csv_path, output_path)
    verify_patient_files(output_path, data_root)