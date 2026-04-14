#!/bin/bash
#SBATCH --job-name=mcp_gpt2
#SBATCH --partition=nvl
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/gpt2_%j.out
#SBATCH --error=logs/gpt2_%j.err

module load gcc/9.3.0
module load python/3.11.9
module load cuda/11.5.0

cd /weka/home/jhasset1/Complexity_profiles

source complexity/bin/activate

# Train GPT-2 Small with test run (1M tokens)
python -m training.train_gpt2 \
    --config configs/training_config.yaml \
    --seed 0 \
    --device cuda \
    --checkpoint_dir checkpoints \
    --test_run
