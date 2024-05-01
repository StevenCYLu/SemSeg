#!/bin/bash

# FILENAME: train.sh

#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2 -c 16 --gpus-per-node=2 --time=4:00:00

nvidia-smi

now=$(date +"%Y%m%d_%H%M%S")
job='canopus'

module load anaconda/2020.11-py38
conda activate ssg

dataset='pascal'
method='canopus'
exp='r101_unsup-512-test'
split='92'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split
gpu_per_nodes=2
nodes=1

mkdir -p $save_path

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
srun --mpi=pmi2 --gres=gpu:$gpu_per_nodes --nodes=$nodes --ntasks-per-node=$gpu_per_nodes --cpus-per-task=16 --job-name=$job \
    python -u $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $1
