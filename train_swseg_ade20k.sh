#!/bin/bash

# Training script for SWSeg method on ADE20K dataset
# Usage: bash train_swseg_ade20k.sh [num_gpus] [master_port] [data_split]
# Example: bash train_swseg_ade20k.sh 2 12345 1_8

# Default parameters
NUM_GPUS=${1:-2}
MASTER_PORT=${2:-12345}
DATA_SPLIT=${3:-1_8}

# Validate data split
if [[ ! -d "splits/ade20k/$DATA_SPLIT" ]]; then
    echo "Error: Data split '$DATA_SPLIT' not found in splits/ade20k/"
    echo "Available splits:"
    ls splits/ade20k/ | grep -E '^1_[0-9]+$'
    exit 1
fi

# Set experiment save path
SAVE_PATH="exp/ade20k/swseg/swseg_ade20k/$DATA_SPLIT"

# Create save directory
mkdir -p "$SAVE_PATH"

# Print configuration
echo "=========================================="
echo "Training SWSeg on ADE20K Dataset"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo "Data Split: $DATA_SPLIT"
echo "Save Path: $SAVE_PATH"
echo "=========================================="

# Run distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    swseg.py \
    --config configs/ade20k.yaml \
    --labeled-id-path splits/ade20k/$DATA_SPLIT/labeled.txt \
    --unlabeled-id-path splits/ade20k/$DATA_SPLIT/unlabeled.txt \
    --save-path "$SAVE_PATH"

echo "Training completed. Results saved to: $SAVE_PATH"