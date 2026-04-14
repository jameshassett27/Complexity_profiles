# MCP Pilot Implementation Status

## What Has Been Set Up

I've organized the pilot implementation with the following components:

### Project Structure
```
Complexity_profiles/
├── configs/                  # Configuration files
│   ├── training_config.yaml  # Training hyperparameters
│   ├── mcp_config.yaml       # MCP mapping configurations
│   └── evaluation_config.yaml # Task A/B evaluation settings
├── models/                   # Model implementations
│   └── dem.py               # Delay Embedding Model (from scratch)
├── data/                     # Data pipelines
│   └── wikitext103.py       # WikiText-103 with GPT-2 BPE tokenizer
├── training/                 # Training scripts
│   └── train_dem.py         # DEM training script
├── mcp/                      # MCP implementation
│   ├── mappings.py          # 4 mapping levels (ridge, kernel ridge, MLPs)
│   ├── metrics.py           # Summary statistics (A, L, K, G)
│   ├── pipeline.py          # Full MCP computation pipeline
│   └── pilot_ablation.py    # Pilot ablation study
├── baselines/               # (to be filled) Baseline metrics
├── extraction/              # (to be filled) Representation extraction
├── analysis/                # (to be filled) Analysis scripts
├── checkpoints/             # Model checkpoints (gitignored)
├── logs/                    # Training logs (gitignored)
└── results/                 # MCP results (gitignored)
```

### Completed Components

1. **Configuration Files** - YAML configs for training, MCP, and evaluation
2. **Data Pipeline** - WikiText-103 with shared GPT-2 BPE tokenizer
3. **DEM Architecture** - Delay Embedding Model built from scratch
4. **MCP Mappings** - All 4 levels (ridge, kernel ridge, 1-layer MLP, 2-layer MLP)
5. **MCP Metrics** - Summary statistics (Asymptote, Linearity Index, Kernel Gain, Learning Gain)
6. **MCP Pipeline** - Full computation with splits, dimensionality reduction, and directional MCP
7. **DEM Training Script** - Training script with cosine LR schedule, early stopping
8. **Pilot Ablation Script** - Framework for testing hyperparameter robustness

### Documentation

- `README.md` - Project overview and directory structure
- `PILOT_PLAN.md` - Detailed Week 1 pilot plan with day-by-day schedule
- `requirements.txt` - Python dependencies
- `setup.sh` - Setup script to initialize the project
- `.gitignore` - Git ignore patterns

## Next Steps for Pilot (Week 1)

### Day 1-2: Data Pipeline ✓ (Already implemented)
- WikiText-103 pipeline is ready and tested
- GPT-2 BPE tokenizer is set up
- Run `python data/wikitext103.py` to test

### Day 2-3: DEM Implementation ✓ (Architecture done, needs training test)
- DEM architecture is implemented in `models/dem.py`
- Training script is ready in `training/train_dem.py`
- **To do**: Run a test training on 1M tokens
  ```bash
  python training/train_dem.py --test_run --device cuda
  ```
- Verify DEM converges (perplexity < 60 or significantly below vocab size)
- Check BPE tokenizer compatibility

### Day 3-5: GPT-2 and Mamba Training (Partial)
**Still needed**:
- Set up nanoGPT for GPT-2 Small training
- Set up mamba-ssm for Mamba-Small training
- Train 1 seed each for ~100M tokens
- Extract representations at layers 4, 8, 12

**Action items**:
```bash
# Clone nanoGPT
git clone https://github.com/karpathy/nanoGPT.git external/nanoGPT

# Install mamba-ssm
pip install mamba-ssm

# Create training wrappers for GPT-2 and Mamba
# (similar to train_dem.py)
```

### Day 5-7: MCP Pipeline + Pilot Ablation ✓ (Pipeline ready, needs real data)
- MCP pipeline is implemented and tested with dummy data
- Pilot ablation framework is ready
- **To do**: Run pilot ablation on real GPT-2 vs Mamba representations
  - Vary MLP hidden sizes: [256, 512, 1024]
  - Vary activation: [GELU, ReLU, SiLU]
  - Vary kernel bandwidth: [0.5×median, median, 2×median]
- Generate Supp Fig S1

### Decision Gate (End of Week 1)
Before proceeding to full training, confirm:
- [ ] DEM converges on WikiText-103
- [ ] MCP profiles show meaningful variation across 4 levels
- [ ] Pilot ablation shows hyperparameter robustness (±0.05 threshold)
- [ ] All pipelines are end-to-end working

## Immediate Action Items

1. **Test the existing components**:
   ```bash
   # Make setup script executable
   chmod +x setup.sh
   
   # Run setup (this will test all components)
   ./setup.sh
   ```

2. **Test DEM training on small subset**:
   ```bash
   python training/train_dem.py --test_run --device cuda
   ```

3. **Set up GPT-2 training infrastructure**:
   - Clone nanoGPT
   - Adapt it for WikiText-103 with your data pipeline
   - Create `training/train_gpt2.py`

4. **Set up Mamba training infrastructure**:
   - Install mamba-ssm
   - Create `training/train_mamba.py`

## Key Design Decisions

1. **Shared Tokenizer**: All models use GPT-2 BPE (critical for fair comparison)
2. **DEM Architecture**: Explicit delay embedding with exponential delays [1, 2, 4, 8, 16, 32]
3. **MCP Levels**: Mixed hierarchy (linear, kernel nonlinear, shallow learned, deep learned)
4. **GPU Acceleration**: MLP training on GPU (thousands of small runs)
5. **Pilot Ablation**: Test hyperparameter robustness before full experiment

## Compute Requirements

- **Pilot**: ~10-15 GPU hours
- **Full experiment**: ~130-200 GPU hours on 1-2 GPUs

With your extensive GPU access on a remote cluster, compute should not be a bottleneck.

## Risk Mitigation

- If DEM fails after 3 days: Drop DEM, proceed with 3 architectures
- If MCP profiles are flat: Rethink function class hierarchy
- Pilot catches fatal issues early

## References

The protocol is based on experimental_protocol_v3_final.md which outlines:
- 4 architectures: GPT-2, Mamba, RWKV, DEM
- 2 tasks: Language Modeling (unstructured), Topic Classification (structured)
- MCP as primary contribution
- Two-task validation for task-dependent representational structure
