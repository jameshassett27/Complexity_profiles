#!/bin/bash
#SBATCH --job-name=mcp_extract
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=jvogels3
#SBATCH --output=logs/extract_mcp_%j.out
#SBATCH --error=logs/extract_mcp_%j.err

cd /home/jhasset1/Complexity_profiles
source complexity/bin/activate

echo "=== Step 1: Extract hidden states ==="
python -m extraction.extract_hidden_states \
    --checkpoint_dir checkpoints \
    --output_dir results/hidden_states \
    --layers 4 8 12 \
    --n_batches 50 \
    --device cuda \
    --models dem gpt2 lstm rwkv

echo "=== Step 2: Run MCP pipeline ==="
python -m mcp.run_mcp \
    --hidden_states_dir results/hidden_states \
    --output_dir results/mcp \
    --layers 4 8 12 \
    --models dem gpt2 lstm rwkv \
    --device cuda
