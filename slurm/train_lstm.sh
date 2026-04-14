#!/bin/bash
#SBATCH --job-name=mcp_lstm
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=jvogels3
#SBATCH --output=logs/lstm_%j.out
#SBATCH --error=logs/lstm_%j.err

cd /weka/home/jhasset1/Complexity_profiles
source complexity/bin/activate

python -m training.train_lstm \
    --config configs/training_config.yaml \
    --seed 0 \
    --device cuda \
    --checkpoint_dir checkpoints \
    --test_run
