# JHU DSAI Cluster Setup Guide

## CUDA and PyTorch Configuration

### Critical Issue: CUDA Version Mismatch

**Problem:**
- Installing PyTorch via pip (`pip install torch`) often installs a version compiled for a newer CUDA version than the cluster supports
- This causes `RuntimeError: The NVIDIA driver on your system is too old` when trying to use CUDA
- The cluster has CUDA 11.5, but pip may install PyTorch compiled for CUDA 11.7, 11.8, or 13.0

**Solution:**
- Use the cluster's PyTorch module instead of installing via pip
- The cluster provides `pytorch/2.5.1` which is pre-configured to work with the cluster's GPU setup

### Correct Module Loading Order

```bash
module load gcc/9.3.0
module load cuda/11.5.0
module load pytorch/2.5.1
```

**Important:** Load modules in this specific order. PyTorch must be loaded after CUDA.

### CUDA Availability

- CUDA is NOT available on the login node
- CUDA IS available on GPU compute nodes (nvl, a100, h100 partitions)
- Always test CUDA on a compute node, not the login node

### Test CUDA on Compute Node

```bash
cat > test_cuda.sh << 'EOF'
#!/bin/bash
#SBATCH --partition=nvl
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=cuda_test.out

module load gcc/9.3.0
module load cuda/11.5.0
module load pytorch/2.5.1

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
EOF

sbatch test_cuda.sh
cat cuda_test.out
```

### Installing Additional Dependencies

After loading the pytorch module, install additional packages:

```bash
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm wandb transformers datasets tokenizers einops pyyaml tensorboard
```

**Note:** You may see dependency conflict warnings from other packages in the cluster's environment (tensorflow, etc.). These can be ignored as they won't affect PyTorch training.

### SLURM Script Configuration

SLURM scripts should load modules in the correct order:

```bash
#!/bin/bash
#SBATCH --partition=nvl
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

module load gcc/9.3.0
module load cuda/11.5.0
module load pytorch/2.5.1

cd /weka/home/jhasset1/Complexity_profiles

python -m training.train_dem \
    --config configs/training_config.yaml \
    --seed 0 \
    --device cuda \
    --checkpoint_dir checkpoints \
    --test_run
```

### Available GPU Partitions

- `nvl`: NVL GPUs (2 idle GPUs often available)
- `a100`: A100 GPUs (currently all busy)
- `h100`: H100 GPUs (currently all busy)
- `l40s`: L40S GPUs

Check availability with: `sinfo -o "%P %A %l %D %T"`

### Why Not Use venv?

- Using the cluster's pytorch module is preferred over creating a venv
- The cluster's module is pre-configured for the cluster's GPU setup
- Avoids CUDA version mismatches
- Fewer installation issues

### Common Errors and Solutions

**Error:** `RuntimeError: The NVIDIA driver on your system is too old`
- **Cause:** PyTorch compiled for newer CUDA version than cluster supports
- **Solution:** Use cluster's pytorch module instead of pip install

**Error:** `torch.cuda.is_available()` returns False on login node
- **Cause:** CUDA not available on login node (normal)
- **Solution:** Test on compute node via SLURM job

**Error:** Module loading order issues
- **Cause:** Wrong module loading sequence
- **Solution:** Always load: gcc → cuda → pytorch

### Summary Checklist

- [ ] Load modules in order: gcc/9.3.0 → cuda/11.5.0 → pytorch/2.5.1
- [ ] Test CUDA on compute node, not login node
- [ ] Use cluster's pytorch module, don't pip install torch
- [ ] Install other dependencies after loading pytorch module
- [ ] Configure SLURM scripts with correct module loads
- [ ] Use nvl partition for faster queue times (often has idle GPUs)
