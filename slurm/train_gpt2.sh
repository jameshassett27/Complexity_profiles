#!/bin/bash
#SBATCH --job-name=mcp_gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/gpt2_%j.out
#SBATCH --error=logs/gpt2_%j.err

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

# Train GPT-2 Small
python training/train_gpt2.py \
    --config configs/training_config.yaml \
    --seed 0 \
    --device cuda \
    --checkpoint_dir checkpoints
