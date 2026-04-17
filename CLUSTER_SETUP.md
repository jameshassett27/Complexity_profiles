# JHU DSAI Cluster Setup

## TL;DR

- PyTorch installed via pip with bundled CUDA 12.1 works fine
- **Jobs MUST specify `#SBATCH --account=jvogels3`** — without it, the job runs but SLURM does not bind `/dev/nvidia*` to the cgroup, so `torch.cuda.is_available()` returns False
- Do NOT use the cluster's `pytorch/2.5.1` module (Lmod error, broken)
- No `module load cuda/*` needed — torch wheels bundle their own CUDA libraries; the driver is enough

## Venv

Located at `/weka/home/jhasset1/Complexity_profiles/complexity/` (Python 3.11). Activate with:
```bash
source complexity/bin/activate
```

PyTorch installed via:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Results in `torch 2.5.1+cu121` with bundled NVIDIA CUDA 12.1 libs. Driver on compute nodes is 560.35.03 (CUDA 12.6), fully compatible.

## SLURM script template

```bash
#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=jvogels3           # REQUIRED for GPU binding
#SBATCH --output=logs/%x_%j.out

cd /weka/home/jhasset1/Complexity_profiles
source complexity/bin/activate

python -m training.train_dem --config configs/training_config.yaml
```

## Partitions

- `a100` — 8x A100 80GB per node
- `h100` — 4x H100 per node
- `nvl` — 4x H100 NVL per node
- `l40s` — 8x L40S per node
- `cpu` — no GPUs

Check availability: `sinfo -o '%P %a %D %T %G'`

## How we debugged "cuda available: False"

Symptoms:
- `nvidia-smi` works inside the job (shows assigned GPU)
- `torch.cuda.is_available()` → False
- `libcuda.cuInit(0)` → error 3 (`CUDA_ERROR_NOT_INITIALIZED`)
- `cat /dev/nvidia0` → `Invalid argument`
- SLURM cgroup: `3:devices:/slurm/.../task_0`

Root cause: job submitted without `--account=jvogels3` lands in a QOS that does not grant GPU device access via the devices cgroup. `nvidia-smi` still works because NVML uses a different kernel path, which made the failure mode confusing.

Fix: add `#SBATCH --account=jvogels3`.

## What does NOT work (skip these rabbit holes)

- `module load pytorch/2.5.1` — Lmod error: "MT:add_property(): system property table has no entry for: environ"
- `module load cuda/11.5.0` — not needed; also too old for torch cu121
- `module load cuda/12.6.3` — module doesn't exist on this cluster
- Installing torch against older CUDA to match `cuda/11.5` module — driver is 12.6, so modern wheels work fine

## Verifying setup on a compute node

```bash
cat > test_cuda.sh << 'EOF'
#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=jvogels3
#SBATCH --output=test_cuda.out

cd /weka/home/jhasset1/Complexity_profiles
source complexity/bin/activate
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
EOF
sbatch test_cuda.sh
```

Note: `torch.cuda.is_available()` is always False on the login node — this is normal, the login node has no GPU. Always test via sbatch/srun.
