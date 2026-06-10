# Quick Command Reference — MRNet Local Learning Branch
# Run all commands from: /Users/jinishrajan/ucd/Medimaging/

# ── Monitor training progress ──────────────────────────────────────────────────
# Paste in a separate Terminal window while training runs:
while true; do clear; ps aux | grep train.py | grep -v grep | awk '{print "CPU: " $3"%" "   MEM: " $4"%   Running for: " $10}'; echo "---"; date; sleep 3; done

# ── Step 3: Train ONE model (1 epoch test — to verify pipeline) ───────────────
.venv/bin/python code/src/train.py \
  --condition acl --plane sagittal --architecture baseline \
  --epochs_phase1 1 --epochs_phase2 1 \
  --data_dir code/src/data/mrnet/ \
  --checkpoint_dir local_learning_hub/checkpoints/

# ── Step 3: Train ONE model (FULL 30 epochs — takes ~3.5hrs on Mac) ──────────
.venv/bin/python code/src/train.py \
  --condition acl --plane sagittal --architecture baseline \
  --epochs_phase1 10 --epochs_phase2 20 \
  --data_dir code/src/data/mrnet/ \
  --checkpoint_dir local_learning_hub/checkpoints/

# ── Step 4: Evaluate ──────────────────────────────────────────────────────────
.venv/bin/python code/src/evaluation.py \
  --condition acl --architecture baseline \
  --checkpoint_dir local_learning_hub/checkpoints/ \
  --data_dir code/src/data/mrnet/ \
  --output_dir local_learning_hub/results/

# ── Step 5: Grad-CAM heatmaps ─────────────────────────────────────────────────
.venv/bin/python code/src/explainability.py \
  --condition acl --plane sagittal --architecture baseline \
  --checkpoint_path local_learning_hub/checkpoints/acl_sagittal_baseline/best_phase2.pth \
  --data_dir code/src/data/mrnet/ \
  --output_dir local_learning_hub/visualisations/

# ── Open output images ────────────────────────────────────────────────────────
open local_learning_hub/results/ACL_baseline_curves.png
open local_learning_hub/visualisations/acl_sagittal/
