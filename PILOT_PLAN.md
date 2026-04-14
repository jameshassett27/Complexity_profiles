# Pilot Implementation Plan (Week 1)

## Overview
The pilot phase (Week 1) is a de-risking phase to validate that:
1. DEM architecture can train and converge on WikiText-103
2. MCP pipeline produces meaningful variation across mapping levels
3. Hyperparameter choices are robust (ablation study)

## Day-by-Day Schedule

### Day 1-2: Data Pipeline Setup
**Goal**: WikiText-103 pipeline with GPT-2 BPE tokenizer

**Tasks**:
- [ ] Download WikiText-103 dataset
- [ ] Set up GPT-2 BPE tokenizer from HuggingFace
- [ ] Implement data loader with sequence length 256
- [ ] Test tokenization on a small batch
- [ ] Verify tokenizer works across all architectures (critical for DEM)

**Deliverables**:
- Working data pipeline in `data/wikitext103.py`
- Tokenized validation set for Task A (10,000 sentences)
- Sanity check: vocab size = 50257

**Success Criteria**:
- Can load and tokenize WikiText-103 without errors
- Tokenization is reproducible with fixed seed
- Batch generation works for training

---

### Day 2-3: DEM Architecture Implementation
**Goal**: Implement Delay Embedding Model from scratch

**Tasks**:
- [ ] Implement DEM architecture in `models/dem.py`
- [ ] Implement exponential delay concatenation with circular buffer
- [ ] Implement temporal mixing blocks (LayerNorm → Linear → GELU → Linear → residual)
- [ ] Test forward pass on dummy input
- [ ] Train on small subset (1M tokens) to verify it learns at all
- [ ] Check BPE tokenizer compatibility

**Architecture Details**:
```
Input tokens
    → BPE embedding (shared tokenizer)
    → Exponential delay concatenation: [e_t, e_{t-1}, e_{t-2}, e_{t-4}, e_{t-8}, e_{t-16}, e_{t-32}]
    → Circular buffer of size 64, O(1) memory
    → Learned linear projection (delay_dim → model_dim) + LayerNorm
    → N feedforward temporal mixing blocks
    → Final LayerNorm → output projection → vocabulary logits
```

**Hyperparameters**:
- model_dim: 512
- n_mixing_blocks: 8
- delay_pattern: [1, 2, 4, 8, 16, 32]
- buffer_size: 64

**Deliverables**:
- Working DEM implementation
- Training script `training/train_dem.py`
- Test run on 1M tokens

**Success Criteria**:
- Forward pass works without errors
- Loss decreases on training subset
- No NaN or instability issues
- Perplexity significantly below vocab size (50257)

**Fallback**: If DEM fails after 3 days of tuning, drop DEM and proceed with 3 architectures

---

### Day 3-5: GPT-2 and Mamba Training (Partial)
**Goal**: Train 1 seed each of GPT-2 and Mamba on ~100M tokens

**Tasks**:
- [ ] Set up nanoGPT/minGPT for GPT-2 Small training
- [ ] Set up mamba-ssm for Mamba-Small training
- [ ] Configure training hyperparameters (lr=3e-4, cosine annealing, etc.)
- [ ] Train GPT-2 seed 0 for ~100M tokens
- [ ] Train Mamba seed 0 for ~100M tokens
- [ ] Extract representations at layers 4, 8, 12 for MCP pilot

**Model Sizes**:
- GPT-2 Small: 12 layers, 8 heads, 512 dim (~20-40M params)
- Mamba-Small: d_model=512, n_layer=12 (~20-40M params)

**Deliverables**:
- Trained GPT-2 seed 0 checkpoint
- Trained Mamba seed 0 checkpoint
- Extracted representations for pilot MCP

**Success Criteria**:
- Both models train without crashes
- Validation perplexity trending downward
- Can extract hidden states at specified layers

---

### Day 5-7: MCP Pipeline + Pilot Ablation
**Goal**: Implement MCP pipeline and run pilot ablation

**Tasks**:
- [ ] Implement 4 mapping levels in `mcp/mappings.py`:
  - Level 1: Ridge regression (closed-form)
  - Level 2: Kernel ridge regression (RBF)
  - Level 3: 1-hidden-layer MLP
  - Level 4: 2-hidden-layer MLP
- [ ] Implement summary statistics in `mcp/metrics.py` (A, L, K, G)
- [ ] Implement full MCP pipeline in `mcp/pipeline.py`
- [ ] Run pilot ablation on GPT-2 vs Mamba (layers 4, 8, 12):
  - Vary MLP hidden sizes: [256, 512, 1024]
  - Vary activation: [GELU, ReLU, SiLU]
  - Vary kernel bandwidth: [0.5×median, median, 2×median]
- [ ] Generate Supp Fig S1 (ablation results)

**Pilot Ablation Details**:
- Architecture pair: GPT-2 seed 0 vs. Mamba seed 0
- Layers: 4, 8, 12 (3 layers)
- For each layer, test hyperparameter variants
- Check robustness: R² variation within ±0.05 is acceptable

**Deliverables**:
- Working MCP pipeline
- Pilot ablation results
- Supp Fig S1

**Success Criteria**:
- MCP profile shows meaningful variation across levels
- Profile shape is robust to hyperparameter choices (±0.05 threshold)
- No crashes or NaN issues

**Decision Gate**:
- ✅ If DEM converges AND MCP shows variation: Proceed to full experiment
- ❌ If DEM fails: Drop DEM, proceed with 3 architectures
- ❌ If MCP is flat (all levels same R²): Rethink function class hierarchy or pivot

---

## Week 1 Decision Gate Checklist

Before proceeding to Weeks 2-3 (full training), confirm:

- [ ] DEM converges on WikiText-103 (perplexity < 60 or significantly below vocab size)
- [ ] MCP profiles show meaningful variation across 4 levels
- [ ] Pilot ablation shows hyperparameter robustness (±0.05 threshold)
- [ ] GPT-2 and Mamba training is stable
- [ ] All pipelines (data, training, extraction, MCP) are end-to-end working
- [ ] Compute budget for full experiment is confirmed (~130-200 GPU-hours)

If any item fails, address before proceeding. The pilot is designed to catch fatal issues early.

---

## Compute Requirements (Pilot)

| Task | GPU Hours |
|------|-----------|
| WikiText-103 pipeline | < 1 |
| DEM test (1M tokens) | 2-3 |
| GPT-2 training (100M tokens) | 3-4 |
| Mamba training (100M tokens) | 3-4 |
| MCP pilot ablation | 1-2 |
| **Total** | **~10-15 GPU hours** |

---

## Key Implementation Notes

1. **Shared Tokenizer**: All models must use the same GPT-2 BPE tokenizer. This is critical for fair comparison, especially for DEM.

2. **GPU Acceleration**: MLP training in MCP should be GPU-accelerated. This will be thousands of small training runs.

3. **Reproducibility**: Fix random seeds for data sampling (Task A: 10,000 sentences from validation set).

4. **Logging**: Use Weights & Biases or TensorBoard for training curves. Save checkpoints regularly.

5. **Error Handling**: Add comprehensive error checking for NaN losses, gradient explosions, etc.

---

## Next Steps After Pilot

If pilot succeeds, proceed to:
- Weeks 2-3: Full training (20 models × 1B tokens)
- Week 4: Representation extraction + MCP computation
- Week 5: Analysis + figures
- Weeks 6-7: Writing
- Week 8: Revision
