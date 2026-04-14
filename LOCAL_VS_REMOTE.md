# Local vs Remote Workflow

## What to Do Locally (NOW)

### 0. Set up virtual environment
```bash
chmod +x setup.sh
./setup.sh
# or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1. Test existing components (no GPU needed, quick)
```bash
# Make sure venv is activated
source venv/bin/activate

# Test each component
python data/wikitext103.py
python models/dem.py
python mcp/mappings.py
python mcp/metrics.py
python mcp/pipeline.py
```

These tests verify the code works without training models. They should complete in <5 minutes.

### 2. Test DEM training (requires local GPU)
```bash
python training/train_dem.py --test_run --device cuda
```
This trains DEM for 1M tokens to verify convergence. Takes ~30 minutes on a GPU.

If you don't have a local GPU, skip this and do it on the remote cluster.

### 3. Create GPT-2 training wrapper
You need to create `training/train_gpt2.py` that:
- Clones/uses nanoGPT
- Uses your WikiText-103 data pipeline (`data/wikitext103.py`)
- Configures GPT-2 Small (12 layers, 8 heads, 512 dim)
- Uses training hyperparameters from `configs/training_config.yaml`

### 4. Create Mamba training wrapper
You need to create `training/train_mamba.py` that:
- Uses mamba-ssm package
- Uses your WikiText-103 data pipeline
- Configures Mamba-Small (d_model=512, n_layer=12)
- Uses training hyperparameters from config

## What to Push to Remote Cluster

Push the **entire project**:
```bash
git add .
git commit -m "Initial MCP pilot implementation"
git push origin main
```

On the remote cluster:
```bash
git clone <your-repo>
cd Complexity_profiles
pip install -r requirements.txt
```

## What to Run on Remote Cluster

### Pilot Phase (Week 1)

**Job 1: DEM test run**
```bash
sbatch slurm/train_dem.sh
```
Modify the script to add `--test_run` flag for 1M tokens.

**Job 2: GPT-2 pilot (after creating train_gpt2.py)**
```bash
sbatch slurm/train_gpt2.sh
```
Modify for ~100M tokens (pilot scale).

**Job 3: Mamba pilot (after creating train_mamba.py)**
```bash
sbatch slurm/train_mamba.sh
```
Modify for ~100M tokens (pilot scale).

**Job 4: Pilot ablation**
After extracting representations from GPT-2 and Mamba:
```bash
python mcp/pilot_ablation.py --device cuda
```

### Full Training (Weeks 2-3)

After pilot succeeds, submit 20 training jobs (4 architectures × 5 seeds):
```bash
for seed in 0 1 2 3 4; do
    sbatch slurm/train_dem.sh --seed $seed
    sbatch slurm/train_gpt2.sh --seed $seed
    sbatch slurm/train_mamba.sh --seed $seed
    # RWKV similar
done
```

## SLURM Setup

Update the SLURM scripts in `slurm/` directory:
- Change `/path/to/Complexity_profiles` to actual path
- Update module loads based on your cluster
- Update partition name if different from `gpu`
- Add environment activation if using conda/venv

## Immediate Action Items

**Local (do now):**
1. Run the 5 component tests (Step 1 above)
2. If you have local GPU, run DEM test (Step 2)
3. Create `training/train_gpt2.py`
4. Create `training/train_mamba.py`

**Remote (after local setup):**
1. Push code to remote
2. Submit DEM test job
3. Submit GPT-2 pilot job
4. Submit Mamba pilot job

## Dependencies

**GPT-2:**
- nanoGPT: `git clone https://github.com/karpathy/nanoGPT.git external/nanoGPT`
- You'll need to adapt nanoGPT to use your WikiText-103 data loader

**Mamba:**
- mamba-ssm: `pip install mamba-ssm`
- You'll need to wrap the mamba-ssm model with your training loop

**RWKV:**
- RWKV implementation (choose one: official repo, or a clean implementation)
- Similar wrapper pattern

## Priority Order

1. **Test existing components** (5 min, local)
2. **Create GPT-2 wrapper** (1-2 hours, local)
3. **Create Mamba wrapper** (1-2 hours, local)
4. **Push to remote** (5 min)
5. **Submit pilot jobs** (remote, 1-2 days runtime)
6. **Run pilot ablation** (remote, after pilot jobs complete)
