#!/bin/bash

# Submits one SLURM job per combination of condition x plane x data_mode x architecture
# using SGD optimizer. Checkpoints are saved to checkpoints_sgd/ to avoid overwriting
# existing AdamW results.
#
# Usage:
#   bash submit_all_sgd.sh

CONDITIONS=(acl meniscus abnormal)
PLANES=(axial coronal sagittal)
DATA_MODES=(uncropped cropped)
ARCHS=(baseline comparative)

TOTAL=$(( ${#CONDITIONS[@]} * ${#PLANES[@]} * ${#DATA_MODES[@]} * ${#ARCHS[@]} ))
echo "Submitting SGD training jobs"
echo "Total jobs: $TOTAL"
echo ""

for ARCH in "${ARCHS[@]}"; do
    for CONDITION in "${CONDITIONS[@]}"; do
        for PLANE in "${PLANES[@]}"; do
            for DATA_MODE in "${DATA_MODES[@]}"; do
                echo "Submitting: $ARCH | $CONDITION | $PLANE | $DATA_MODE | SGD"
                sbatch \
                    --export=ALL,CONDITION=$CONDITION,PLANE=$PLANE,DATA_MODE=$DATA_MODE,ARCH=$ARCH \
                    submit_sgd_job.sh
            done
        done
    done
done

echo ""
echo "All $TOTAL jobs submitted. Check status with: squeue -u \$USER"
