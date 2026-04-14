#!/bin/bash
#SBATCH --job-name=mcp_dem
#SBATCH --partition=nvl
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/dem_%j.out
#SBATCH --error=logs/dem_%j.err

module load gcc/9.3.0
module load python/3.11.9
module load cuda/11.5.0

cd /weka/home/jhasset1/Complexity_profiles

source complexity/bin/activate

# Train DEM with test run (1M tokens)
python -m training.train_dem \
    --config configs/training_config.yaml \
    --seed 0 \
    --device cuda \
    --checkpoint_dir checkpoints \
    --test_run
