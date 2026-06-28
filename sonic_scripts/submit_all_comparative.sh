#!/bin/bash

# Submits one SLURM job per combination of condition x plane x data_mode
# for the comparative model (xrv_dense by default).
#
# Usage:
#   bash submit_all_comparative.sh            # runs xrv_dense (medical comparison)
#   bash submit_all_comparative.sh resnet50   # runs resnet50 (ImageNet comparison)

COMP_ARCH=${1:-xrv_dense}

CONDITIONS=(acl meniscus abnormal)
PLANES=(axial coronal sagittal)
DATA_MODES=(uncropped cropped)

echo "Submitting comparative jobs with arch: $COMP_ARCH"
echo "Total jobs: $(( ${#CONDITIONS[@]} * ${#PLANES[@]} * ${#DATA_MODES[@]} ))"
echo ""

for CONDITION in "${CONDITIONS[@]}"; do
    for PLANE in "${PLANES[@]}"; do
        for DATA_MODE in "${DATA_MODES[@]}"; do
            echo "Submitting: $CONDITION | $PLANE | $DATA_MODE | $COMP_ARCH"
            sbatch \
                --export=ALL,CONDITION=$CONDITION,PLANE=$PLANE,DATA_MODE=$DATA_MODE,COMP_ARCH=$COMP_ARCH \
                code/src/submit_comparative_job.sh
        done
    done
done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
