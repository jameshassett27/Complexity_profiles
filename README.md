# Mapping Complexity Profiles (MCP) - Pilot Implementation

## Project Overview
This project implements Mapping Complexity Profiles, a diagnostic tool for characterizing representational similarity beyond single-number metrics. The pilot validates the method across 4 sequence architectures (GPT-2, Mamba, RWKV, Delay Embedding Model) on two tasks (Language Modeling and Topic Classification).

## Directory Structure
```
.
├── configs/               # Configuration files for training and MCP
├── data/                  # Dataset storage (WikiText-103, AG News, SST-2)
├── models/                # Model implementations
│   ├── dem.py            # Delay Embedding Model (from scratch)
│   ├── gpt2_wrapper.py   # GPT-2 wrapper around nanoGPT/minGPT
│   ├── mamba_wrapper.py  # Mamba wrapper around mamba-ssm
│   └── rwkv_wrapper.py   # RWKV wrapper
├── training/              # Training scripts
│   ├── train_dem.py
│   ├── train_gpt2.py
│   ├── train_mamba.py
│   └── train_rwkv.py
├── mcp/                   # MCP implementation
│   ├── mappings.py       # 4 mapping levels (ridge, kernel ridge, MLPs)
│   ├── metrics.py        # Summary statistics (A, L, K, G)
│   └── pipeline.py       # Full MCP computation pipeline
├── baselines/            # Baseline metrics (CKA, RSA, Procrustes, SVCCA)
├── extraction/            # Representation extraction for Tasks A & B
├── analysis/              # Analysis scripts and figure generation
├── checkpoints/          # Model checkpoints (gitignored)
├── logs/                 # Training logs (gitignored)
└── results/              # MCP results and figures (gitignored)
```

## Pilot Phase (Week 1) Goals
1. **Day 1-2**: WikiText-103 pipeline with GPT-2 BPE tokenizer
2. **Day 2-3**: Implement DEM architecture, test on small subset (1M tokens)
3. **Day 3-5**: Train 1 seed each of GPT-2 and Mamba (~100M tokens)
4. **Day 5-7**: Implement MCP pipeline, run pilot ablation
5. **Decision gate**: Does DEM converge? Does MCP show meaningful variation?

## Key Dependencies
- PyTorch
- HuggingFace transformers (for GPT-2 tokenizer)
- mamba-ssm (for Mamba)
- nanoGPT or minGPT (for GPT-2 training)
- scikit-learn (for ridge regression, PCA)
- AG News and SST-2 datasets (HuggingFace datasets)

## Compute Requirements (Pilot)
- ~10-15 GPU hours
- 1 GPU sufficient for pilot phase
- Full experiment: ~130-200 GPU hours on 1-2 GPUs

## Risk Mitigation
- If DEM fails to converge after 3 days: drop DEM, proceed with 3 architectures
- If MCP profiles are flat (all levels same R²): rethink function class hierarchy
- Pilot ablation tests hyperparameter robustness before full experiment
