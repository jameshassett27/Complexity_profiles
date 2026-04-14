#!/bin/bash
#SBATCH --job-name=mcp_mamba
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/mamba_%j.out
#SBATCH --error=logs/mamba_%j.err

module load python/3.10
module load cuda/11.8

cd /path/to/Complexity_profiles

# Activate environment (choose one)
# Option 1: venv
source venv/bin/activate

# Option 2: conda (if available)
# module load anaconda
# conda activate mcp_env

# Install dependencies if needed
# pip install -r requirements.txt

# Train Mamba-Small
python training/train_mamba.py \
    --config configs/training_config.yaml \
    --seed 0 \
    --device cuda \
    --checkpoint_dir checkpoints
