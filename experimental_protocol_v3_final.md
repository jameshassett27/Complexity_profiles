# Mapping Complexity Profiles: A Diagnostic Tool for Representational Similarity
## Experimental Protocol v3.0 (Final)

**Working title**: "Mapping Complexity Profiles: Characterizing Representational Similarity Beyond Single-Number Metrics"

**Framing**: Methods contribution. MCP is the primary novelty. Two-task comparison is the key validation. Information surface theory motivates (with explicit citations) but is not the claim.

---

## 1. Elevator Pitch

CKA tells you two systems are "0.82 similar." But what does that mean? Are they linearly related with noise, or deeply nonlinearly entangled with coincidentally high kernel alignment? We propose Mapping Complexity Profiles (MCP) — a curve characterizing alignment as a function of mapping expressiveness — that reveals qualitative distinctions between architecture pairs invisible to scalar metrics. We validate MCP across five sequence architectures spanning recurrent, attention, state-space, linear attention, and explicit delay-embedding paradigms, showing it captures task-dependent representational structure that CKA, nonlinear CKA, and RSA systematically miss.

---

## 2. Architecture Zoo

### 2.1 Models Trained from Scratch

| Model | Type | Mechanism | Target Size | Why Include |
|-------|------|-----------|------------|-------------|
| LSTM-Small | Recurrent | Gated recurrence | 10-20M | Maximal compression baseline, trains fast, no installation issues |
| GPT-2 Small (custom) | Transformer | Multi-head self-attention | 20-40M | Attention baseline, most studied |
| Mamba-Small | State-space model | Selective state spaces | 20-40M | Linear-time SSM, extremely timely |
| RWKV-Small | Linear attention / recurrent | Time-mixing + channel-mixing | 20-40M | Recurrent alternative |
| Delay Embedding Model (DEM) | Explicit phase-space reconstruction | Exponential delay coordinates + learned projection | 10-20M | No attention, grounded in dynamical systems theory. Most novel inclusion. |

**Seeds per architecture**: 5
**Total training runs**: 25

### 2.2 The Delay Embedding Model (DEM) — Built from Scratch

Rather than using any existing implementation, we construct a minimal delay-embedding sequence model from first principles, motivated by theoretical work connecting attention mechanisms to delay-coordinate reconstruction (Ostrow, Eisen & Fiete, 2024; Takens, 1981) and the broader framing of network depth as dynamical evolution (Chen et al., 2018).

**Architecture**:
```
Input tokens
    → BPE embedding (shared tokenizer with all other models)
    → Exponential delay concatenation: [e_t, e_{t-1}, e_{t-2}, e_{t-4}, e_{t-8}, e_{t-16}, e_{t-32}]
    → Circular buffer of size 64, O(1) memory
    → Learned linear projection (delay_dim → model_dim) + LayerNorm
    → N feedforward temporal mixing blocks (each: LayerNorm → Linear → GELU → Linear → residual)
    → Final LayerNorm → output projection → vocabulary logits
```

**Key design decisions**:
- Uses the same BPE tokenizer as all other models (critical for fair comparison)
- No attention, no recurrence, no state-space mechanism — context integration is purely through delay coordinates
- Exponential delay spacing captures multi-scale temporal structure (local syntax at delay=1, clause structure at delay=4-8, discourse at delay=16-32)
- Fixed memory footprint regardless of sequence length
- ~10-20M parameters depending on model_dim and number of mixing blocks

**What we cite**: Takens (1981) for the delay embedding theorem. Ostrow, Eisen & Fiete (2024) for the theoretical connection between attention and delay embedding in sequence models. Chen et al. (2018) for Neural ODEs and the depth-as-time perspective. We do NOT cite any unpublished or self-published implementations.

**What we claim**: "Motivated by recent theoretical work suggesting that attention mechanisms implicitly perform delay-coordinate reconstruction of latent dynamics (Ostrow et al., 2024), we construct a minimal architecture that performs this reconstruction explicitly, to test whether representational geometry is preserved when the context-integration mechanism is fundamentally different."

### 2.3 Pre-trained Reference Model (inference only)

| Model | Size | Why Include |
|-------|------|-------------|
| Pythia-410M (step 143000, final checkpoint) | 410M | Same family as GPT-2 but ~10× larger, trained on The Pile. "Closer to the surface" reference. Single checkpoint, no seed variation. |
| Pythia-160M (step 143000) | 160M | Intermediate scale reference. Helps distinguish architecture effects from scale effects. |

---

## 3. Training Data

### Primary Training Corpus: WikiText-103

- ~103M tokens, Wikipedia articles
- Standard LM benchmark, widely cited
- GPT-2 BPE tokenizer (50257 vocab) shared across ALL models including DEM

### Training Setup

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Objective | Next-token prediction (cross-entropy) | Standard autoregressive LM |
| Sequence length | 256 tokens | Manageable for DEM delay buffer; consistent across architectures |
| Tokenizer | GPT-2 BPE (HuggingFace) | Shared across all models for fair comparison |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) | Standard |
| Learning rate | 3e-4 with cosine annealing to 3e-5 | Standard for small LMs |
| Weight decay | 0.1 | Standard |
| Batch size | ~64 sequences (adjust per model for comparable tokens/step) | |
| Gradient clipping | 1.0 | |
| Training tokens | ~1B (≈10 epochs over WikiText-103) | |
| Early stopping | Patience 5 epochs on validation perplexity | |

**Convergence target**: All models should achieve validation perplexity < 60 on WikiText-103. If DEM fails to reach this, we report its actual perplexity and note it — the representation comparison is still valid as long as the model has learned *something* non-trivial (perplexity significantly below vocab size of 50257).

---

## 4. Two Tasks, One Set of Models

### Task A: Language Modeling Representations (unstructured)

**Evaluation corpus**: 10,000 sentences sampled from WikiText-103 validation set (fixed random seed, same sentences for all models).

**Extraction**: Pass each sentence through each model. At each layer, take hidden state at the final token position. Result: matrix [10000, d_layer] per model per layer.

These representations have no imposed class structure — their geometry reflects whatever the model learned about language.

### Task B: Topic Classification Representations (structured)

**Primary dataset**: **AG News** (4 classes: World, Sports, Business, Sci/Tech; 120,000 train / 7,600 test).

AG News is preferred over 20 Newsgroups for several reasons: documents are shorter (1-2 sentences, minimal truncation issues at 256 tokens), larger dataset (more stable MCP estimates), cleaner class structure, and more commonly used in recent NLP benchmarks.

**Supplementary dataset** (robustness check): **SST-2** (binary sentiment, Stanford Sentiment Treebank). Very different structure from AG News — if MCP task-dependence holds across both classification tasks, the finding is more robust.

**Method**: Pass each document through each FROZEN model (no fine-tuning). Extract hidden state at final token, each layer. Result: matrix [n_docs, d_layer] per model per layer, plus class labels.

### What the Two-Task Design Tests

Compute MCP and all baseline metrics for the same architecture pair on both tasks. The key demonstration: if CKA gives similar values but MCP shapes differ across tasks, MCP captures task-dependent structure that scalar metrics miss.

---

## 5. Mapping Complexity Profiles — Formal Method

### 5.1 Definition

Given two representation matrices **X** ∈ ℝ^{n×d₁} and **Y** ∈ ℝ^{n×d₂} over the same n inputs, the Mapping Complexity Profile is:

**MCP(X, Y)** = [ R²₁, R²₂, R²₃, R²₄ ]

where R²_k is the multivariate coefficient of determination (mean per-dimension R², averaged across output dimensions of Y) for mapping f_k, evaluated on held-out data.

### 5.2 Mapping Levels

| Level | Mapping | Specification | Complexity Class |
|-------|---------|--------------|-----------------|
| 1 | Ridge regression | α=1.0, closed-form | Linear |
| 2 | Kernel ridge regression (RBF) | α=1.0, bandwidth=median heuristic | Kernel nonlinear |
| 3 | 1-hidden-layer MLP | 512 units, GELU, weight decay 1e-4 | Shallow learned nonlinear |
| 4 | 2-hidden-layer MLP | 1024→512, GELU, weight decay 1e-4 | Deep learned nonlinear |

**Revision from v2**: Replaced the 4-level all-MLP hierarchy with a mixed hierarchy that includes kernel ridge regression (Level 2). This addresses the reviewer concern about arbitrary MLP sizing by grounding the levels in distinct function classes:
- Level 1: Linear maps (hypothesis: representations differ by rotation/scaling)
- Level 2: Fixed nonlinear expansion via RBF kernel (hypothesis: smooth nonlinear relationship)
- Level 3: Learned shallow nonlinearity (hypothesis: structured nonlinear relationship)
- Level 4: Learned deep nonlinearity (hypothesis: complex, hierarchical relationship)

Each level represents a qualitatively different function class with increasing approximation capacity, not just "more parameters." This is more defensible than arbitrary MLP depth increments.

**Theoretical framing**: The levels correspond roughly to increasing VC dimension / approximation capacity. Level 1 tests linear decodability (well-established in probing literature). Level 2 tests kernel-smoothness (connected to CKA theory — RBF-CKA is essentially the scalar summary of what Level 2 captures). Levels 3-4 test whether learned, adaptive nonlinearities unlock additional structure beyond what fixed kernels capture.

### 5.3 MLP Hyperparameters (Justified)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Hidden units (L3) | 512 | Matches typical probing literature; wide enough to approximate smooth functions |
| Hidden units (L4) | 1024→512 | Standard "funnel" architecture; 2× width of L3 to ensure strictly greater capacity |
| Activation | GELU | Smooth, avoids dead neurons (unlike ReLU); standard in modern transformers |
| Optimizer | AdamW, lr=5e-4 | Lower than model training; appropriate for small probe networks |
| Weight decay | 1e-4 | Light regularization to prevent memorization of mapping |
| Batch size | 512 | Large batches for stable gradient estimates on regression task |
| Early stopping | Patience 20 epochs on validation MSE | Prevents overfitting to train split |
| Max epochs | 300 | |
| Dropout | 0.1 between hidden layers | Additional regularization |

### 5.4 Pilot Ablation (Week 1)

Before running the full experiment, conduct a small ablation on 1-2 architecture pairs (e.g., GPT-2 seed 1 vs. Mamba seed 1, at 2-3 layers):
- Vary MLP hidden sizes: [256, 512, 1024] for Level 3
- Vary activation: [GELU, ReLU, SiLU]
- Vary kernel bandwidth for Level 2: [0.5×median, median, 2×median]
- Verify that MCP profile shape is robust to these choices (relative ordering of levels preserved, summary statistics stable within ±0.05)
- Report ablation results in supplementary materials

### 5.5 Fitting Procedure

For each (X, Y) pair:
1. **Split**: 60% train / 20% validation / 20% test (stratified by class labels for Task B)
2. **Dimensionality**: If d > 1024, apply PCA retaining 95% variance. Report results BOTH with and without PCA (reviewer concern about discarding nonlinear structure). Consider random projection as an alternative linear reduction that preserves distances better.
3. **Level 1**: Closed-form ridge regression on train set. R² on test set.
4. **Level 2**: Kernel ridge regression with RBF kernel (bandwidth = median pairwise distance in train set). R² on test set.
5. **Levels 3-4**: Train MLP on train set with early stopping on validation set. R² on test set.
6. **Repeat**: 5 random splits. Report mean ± std.
7. **Directionality**: Compute MCP(X→Y) and MCP(Y→X). Report both.

### 5.6 Multivariate R² (Explicit Definition)

For output Y ∈ ℝ^{n×d₂} and prediction Ŷ = f(X):

R² = 1 - (Σᵢ Σⱼ (Yᵢⱼ - Ŷᵢⱼ)²) / (Σᵢ Σⱼ (Yᵢⱼ - Ȳⱼ)²)

This is the total fraction of variance explained across all dimensions jointly. Equivalent to 1 - (total residual SS / total SS). Single number per mapping.

### 5.7 Summary Statistics

| Statistic | Formula | Interpretation |
|-----------|---------|---------------|
| **Asymptote (A)** | R²₄ | Total shared information accessible by any mapping |
| **Linearity Index (L)** | R²₁ / max(R²₄, ε) | Fraction of shared information that is linearly accessible. L≈1: linear relationship. L≈0: nonlinear. |
| **Kernel Gain (K)** | (R²₂ - R²₁) / max(R²₄ - R²₁, ε) | Fraction of nonlinear structure captured by fixed (non-learned) kernels |
| **Learning Gain (G)** | (R²₄ - R²₂) / max(R²₄ - R²₁, ε) | Fraction of nonlinear structure requiring learned (adaptive) mappings |

Note: K + G ≈ 1 by construction (they partition the nonlinear component). If K is high, the nonlinearity is "smooth" and kernel methods suffice. If G is high, the nonlinearity is "structured" and requires learned features.

### 5.8 Controls

| Control | Method | Expected | Purpose |
|---------|--------|----------|---------|
| **Self-mapping** | MCP(X, X + small noise) | R² ≈ 1.0 all levels | Ceiling |
| **Same-arch diff-seed** | MCP(GPT2-s1, GPT2-s2) | High R², high L | Initialization noise baseline |
| **Shuffled** | MCP(X, permute_rows(Y)) | R² ≈ 0 all levels | Floor / null distribution |
| **Reversed** | MCP(Y→X) vs MCP(X→Y) | Similar if symmetric | Asymmetry diagnostic |
| **Random baseline** | MCP(random_matrix, Y) | R² ≈ 0 | Confirms signal is from representations, not dimensionality |
| **Permutation test** | Shuffle 1000×, compute null distribution of each R²_k | p-values for each level | Formal significance |

---

## 6. Baseline Metrics (Comprehensive)

| Metric | Type | Implementation | Why Include |
|--------|------|---------------|-------------|
| **Linear CKA** | Scalar, linear | HSIC-based, standard implementation | Current gold standard |
| **RBF CKA** (σ=median) | Scalar, nonlinear | Gaussian kernel CKA | Addresses reviewer concern: captures some nonlinearity. Key comparison — does MCP Level 2 reveal more than RBF-CKA? |
| **RSA** (Spearman) | Scalar, rank-based | Spearman correlation of RDM upper triangles | Neuroscience standard |
| **Procrustes distance** | Scalar, geometric | After centering + optimal rotation | Geometric alignment |
| **Effective dimensionality** | Scalar, per-system | Participation ratio of eigenspectrum | Not similarity, but tests shared manifold dimension |
| **SVCCA** | Scalar, subspace | Singular vector CCA (Raghu et al. 2017) | Earlier method; included for completeness |

---

## 7. Analysis Plan (5-Section Paper Narrative)

### Result 1: MCP Reveals Structure Scalar Metrics Miss

For all architecture pairs at best-matching layers (defined as: the layer pair maximizing R²₄) on Task A:
- Compute all scalar metrics + full MCP
- **Key test**: Find pairs where linear CKA AND RBF-CKA are within 0.05 of each other, but MCP Linearity Index (L) or Kernel Gain (K) differs by >0.15
- **Key figure (Fig 2)**: Scatter of CKA vs. Linearity Index for all pairs. Dispersion = MCP adds information.
- **Statistical test**: Spearman correlation between CKA rankings and L rankings. If ρ < 0.8, the metrics disagree substantively.

### Result 2: Convergence Hierarchy

Rank architecture pairs by Linearity Index. Test predicted ordering:
1. Same architecture, different seed (highest L)
2. Same mechanism family (GPT-2 vs. Pythia — both attention, different scale; LSTM vs. LSTM — both recurrent)
3. Different mechanism, same paradigm (GPT-2 vs. Mamba — both learned on WikiText-103; LSTM vs. RWKV — both recurrent)
4. Fundamentally different mechanism (GPT-2 vs. DEM — attention vs. delay embedding; LSTM vs. GPT-2 — recurrent vs. attention)

**Statistical test**: Kruskal-Wallis test across hierarchy levels, followed by pairwise Mann-Whitney with Bonferroni correction.

### Result 3: Task-Dependent MCP Shapes (Key Validation)

For each architecture pair, compare MCP on Task A (LM) vs. Task B (AG News classification):
- **Key figure (Fig 4)**: Side-by-side MCP curves, same pair, both tasks
- **Prediction**: Classification MCPs should be more linear (higher L) than LM MCPs, because classification imposes low-dimensional class structure that constrains representations
- **Statistical test**: Paired Wilcoxon signed-rank test on L values across all architecture pairs (Task A vs. Task B). Permutation test on MCP shape difference (Euclidean distance between 4-point profiles).
- **Robustness**: Repeat with SST-2 as Task B'. If the effect holds across both AG News and SST-2, it's robust to task choice.

### Result 4: Scale Effect (Pythia as Reference)

Compare MCP from Pythia-410M to each small model vs. small-model-to-small-model pairs:
- **Test**: Is Pythia→small-GPT2 more linear than small-GPT2→Mamba? (Same architecture at larger scale vs. different architecture at same scale)
- Include Pythia-160M as intermediate point
- **Key figure**: Linearity Index as a function of "distance" (same arch diff scale, diff arch same scale, diff arch diff scale)

### Result 5: Delay Embedding Model Analysis

Dedicated analysis of where DEM falls in the representational landscape:
- Is DEM more similar to attention models or to SSMs?
- **Key figure (Fig 5)**: MCP from DEM manifold state to each layer of GPT-2. Which transformer depth does explicit delay embedding most resemble?
- Report: "Consistent with theoretical predictions that attention implicitly performs delay-coordinate reconstruction (Ostrow et al., 2024), we find that DEM representations at the manifold projection stage are most aligned with GPT-2 layer [X], with a Linearity Index of [Y]." OR: "Contrary to this prediction, DEM representations show low alignment with all transformer layers, suggesting that explicit and implicit delay embedding produce qualitatively different representational geometries."
- Either result is interesting and publishable.

---

## 8. Figures Outline

| Figure | Content | Section |
|--------|---------|---------|
| **Fig 1** | Method schematic: two systems → four mapping levels → MCP curve → summary statistics | Methods |
| **Fig 2** | CKA vs. Linearity Index scatter (all pairs, Task A). Key: shows MCP ≠ CKA. | Result 1 |
| **Fig 3** | Linearity Index heatmap, architectures × architectures, Task A | Result 2 |
| **Fig 4** | Side-by-side MCP curves: same pair on Task A vs. Task B. The money figure. | Result 3 |
| **Fig 5** | MCP from DEM to each GPT-2 layer | Result 5 |
| **Fig 6** | Effective dimensionality comparison across architectures | Supporting |
| **Supp Fig S1** | Pilot ablation: MCP robustness to hyperparameter choices | Methods validation |
| **Supp Fig S2** | Full layer×layer MCP matrices for all pairs | Completeness |
| **Supp Fig S3** | PCA vs. no-PCA comparison | Robustness |
| **Supp Fig S4** | SST-2 replication of Task B results | Robustness |
| **Supp Fig S5** | Training curves for all 20 models | Reproducibility |

---

## 9. Explicit Theory Citations (Reviewer Concern)

The paper's theoretical motivation draws on the following established work, cited explicitly in the introduction:

| Concept | Citation | How We Use It |
|---------|----------|---------------|
| Representational convergence across architectures | Huh, Cheung, Wang & Isola (2024), "The Platonic Representation Hypothesis," ICML | Motivates the question: if representations converge, what is the nature of that convergence? |
| Task constraints narrow the solution space | Cao & Yamins (2024), "Contravariance principle," Cognitive Systems Research | Motivates the two-task prediction: harder/more constrained tasks should produce more aligned representations |
| Linear vs. nonlinear brain-model mappings | Kriegeskorte (2015), "Deep neural networks: a new framework for modeling biological vision and brain information processing" | Prior discussion of whether brain-model alignment should be measured with linear or nonlinear mappings. MCP formalizes this as a spectrum. |
| Model stitching as functional similarity | Bansal, Nakkiran & Barak (2021), NeurIPS | Closest methodological precedent. We position MCP as complementary: stitching measures task performance, MCP measures representational structure. |
| CKA as representation comparison | Kornblith, Norouzi, Lee & Hinton (2019), ICML | The metric we're extending beyond. |
| Attention as implicit delay embedding | Ostrow, Eisen & Fiete (2024), arXiv (MIT) | Theoretical motivation for the DEM architecture. |

**Key framing sentence**: "We do not claim to test a specific theory of representational convergence. Rather, we provide a diagnostic tool — Mapping Complexity Profiles — that operationalizes questions raised by recent theoretical work (Huh et al., 2024; Cao & Yamins, 2024) about the nature and structure of similarity between learned representations."

---

## 10. Timeline

### Week 1: Pilot + Infrastructure (DE-RISKING PHASE)

- [ ] **Day 1-2**: Set up WikiText-103 pipeline with GPT-2 BPE tokenizer
- [ ] **Day 2-3**: Implement DEM architecture from scratch. Test that it trains at all on a small subset (1M tokens). Verify BPE compatibility.
- [ ] **Day 3-5**: Train 1 seed each of GPT-2 and Mamba on WikiText-103 (partial, ~100M tokens). Extract representations at 2-3 layers.
- [ ] **Day 5-7**: Implement MCP pipeline. Run pilot ablation (GPT-2 vs. Mamba, 2-3 layers, vary MLP sizes/activations/kernel bandwidths). Generate Supp Fig S1.
- [ ] **End of Week 1 decision gate**: Does DEM converge? Does MCP show meaningful variation across levels? If both yes, proceed. If DEM fails, drop it. If MCP is flat (all levels give same R²), rethink the method — this would be a fatal problem.

### Weeks 2-3: Full Training

- [ ] Train all 25 models (5 architectures × 5 seeds) on WikiText-103, full 1B tokens
- [ ] Parallelize: 2-4 models per GPU simultaneously if memory allows
- [ ] Track validation perplexity for all models. Log training curves.
- [ ] Download Pythia-410M and Pythia-160M from HuggingFace.

### Week 4: Extraction + MCP Computation

- [ ] Extract Task A representations: 10,000 sentences × 27 models × all layers
- [ ] Extract Task B representations: AG News test set × 27 frozen models × all layers
- [ ] Extract Task B' representations: SST-2 × 27 frozen models × best-matching layers only
- [ ] Run MCP for all within-architecture seed pairs (controls)
- [ ] Run MCP for all cross-architecture pairs, Task A (Results 1, 2, 5)
- [ ] Run MCP for all cross-architecture pairs, Task B (Result 3)
- [ ] Compute all baseline metrics (CKA, RBF-CKA, RSA, Procrustes, SVCCA, effective dim)
- [ ] **GPU-accelerate MLP training stage** (run MLP probes on GPU, not CPU — this is thousands of small training runs)

### Week 5: Analysis + Figures

- [ ] Generate all primary and supplementary figures
- [ ] Run all statistical tests (permutation tests, Kruskal-Wallis, paired Wilcoxon)
- [ ] Identify strongest CKA-vs-MCP divergence examples
- [ ] Identify strongest task-dependent MCP shape changes
- [ ] DEM layer-matching analysis
- [ ] Write results section

### Weeks 6-7: Writing

- [ ] Introduction: Limitations of scalar metrics → MCP proposal → theoretical motivation (PRH, Contravariance) → contributions
- [ ] Related work: CKA, RSA, model stitching, probing, PRH, attention-as-delay-embedding
- [ ] Methods: MCP formal definition, architecture descriptions, training, evaluation
- [ ] Results: 5-section narrative
- [ ] Discussion: Implications for representation comparison, brain-model alignment (future), dynamical systems perspective (future). Limitations.
- [ ] Supplementary materials

### Week 8: Revision

- [ ] Advisor feedback
- [ ] Code cleaning + reproducibility package
- [ ] Venue selection and formatting

---

## 11. Compute Budget

| Task | Hardware | Hours |
|------|----------|-------|
| Pilot (Week 1) | 1 GPU | ~10-15 |
| Train 20 models (LSTM, GPT-2, Mamba, RWKV × 5 seeds) | 1-2 GPUs | ~65-85 |
| Train 5 DEM models | 1 GPU (or CPU) | ~15-30 |
| Representation extraction (all models, both tasks) | 1 GPU | ~4-6 |
| Pythia inference | 1 GPU (16GB+ VRAM) | ~3-4 |
| MCP computation (all pairs × layers × tasks × splits) | **GPU-accelerated** | ~30-50 |
| Baseline metrics | CPU | ~5-10 |
| **Total** | | **~135-205 GPU-hours** |

Feasible on 1-2 GPUs over 3-4 weeks of wall time.

---

## 12. Risk Register (Final)

| Risk | Likelihood | Impact | Mitigation | Decision Point |
|------|-----------|--------|------------|---------------|
| DEM doesn't converge on WikiText-103 | Medium | Medium | Try smaller model, simpler corpus (subset). If fails after 3 days of tuning, drop DEM. Paper is still strong with 3 architectures. | End of Week 1 |
| MCP levels all give same R² (flat profile) | Low | Fatal | Would mean mapping complexity doesn't vary, killing the method. Pilot in Week 1 catches this early. If flat: rethink function class hierarchy or pivot to a different contribution. | End of Week 1 |
| CKA and MCP give identical rankings everywhere | Low-Med | High | Focus on profile *shape* differences and task-dependence even if rankings agree. Include RBF-CKA as stronger baseline. | Week 5 |
| Mamba installation failure (CUDA version incompatibility) | Low-Med | Low | Train Mamba on Colab Pro if cluster CUDA < 11.6. LSTM provides trivial fallback on cluster. | Week 2 |
| Task B truncation artifacts (AG News) | Low | Low | AG News documents are short (1-2 sentences), truncation is minimal. SST-2 backup is single-sentence. | Week 4 |
| PCA discards nonlinear structure | Med | Med | Report WITH and WITHOUT PCA. Try random projection as alternative. If results diverge, discuss in limitations. | Week 5 |
| Compute for MCP exceeds estimate | Med | Low | GPU-accelerate MLP stage. Subsample to 5,000 sentences if needed (report robustness to sample size). | Week 4 |
| Reviewers say "this is just probing with extra steps" | Med | Med | Emphasize: probing tests decodability of specific *properties*. MCP characterizes the full *representational relationship* between two systems. Different questions. Position relative to Kriegeskorte's linear/nonlinear mapping discussion. | Writing phase |

---

## 13. Target Venues

| Venue | Type | Deadline (approx) | Fit |
|-------|------|-------------------|-----|
| **ICLR 2027** | Top conference | Sep 2026 | Feasible. Methods + empirical. Strong fit for representation learning track. |
| **TMLR** | Rolling journal | Anytime | Great for methods papers. No deadline pressure. Respected. |
| **NeurIPS 2026 Workshop** | Workshop | Sep 2026 | Lower bar, good for early visibility. Submit 4-page version. |
| **ICML 2027** | Top conference | Jan 2027 | Feasible with more time for revisions. |
| **Neural Computation** | Journal | Rolling | Methods + theory. Slower but high-quality venue. |

**Recommended strategy**: Submit workshop version to NeurIPS 2026 (4 pages, preliminary results). Use feedback to strengthen for ICLR 2027 or TMLR full submission.

---

## 14. Roadmap Beyond Paper 1

| Paper | Builds On | Key Addition |
|-------|-----------|-------------|
| **Paper 2** | MCP method | Apply MCP to brain-model alignment (NSD for vision, Narratives dataset for language). First use of MCP with biological system. |
| **Paper 3** | DEM + dynamics | Trajectory MCP — compare layer-wise representation trajectories, not static snapshots. Phase-space reconstruction framework. |
| **Paper 4** | LPN connection | Program-space MCP. Compare latent program geometries across architectures. |
| **Paper 5** | Connectomics | Does structural connectivity predict MCP between brain regions? Your lab's unique angle. |

---

## 15. Pre-Registration Checklist

Before starting Week 1, confirm:

- [ ] GPU access secured (how many, what type, for how long?)
- [ ] WikiText-103 downloaded and tokenized
- [ ] LSTM implementation set up and tested (trivial - standard PyTorch)
- [ ] nanoGPT / minGPT codebase set up and tested
- [ ] Mamba-ssm package installed and tested (or alternative plan if unavailable)
- [ ] RWKV implementation identified and tested
- [ ] DEM architecture coded (even if untested on real data)
- [ ] Pythia models downloadable from HuggingFace (verify VRAM requirements)
- [ ] AG News and SST-2 datasets downloaded
- [ ] MCP pipeline skeleton coded (ridge regression + MLP training loop)
- [ ] Advisor has signed off on the experimental plan

---

*Protocol v3.0 (Final). Language-first. MCP as primary contribution. Delay Embedding Model built from scratch with legitimate citations. Two-task validation with statistical rigor. Pilot ablation for de-risking. All reviewer feedback incorporated.*
