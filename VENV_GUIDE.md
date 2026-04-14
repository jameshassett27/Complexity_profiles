# Virtual Environment Setup

## Quick Setup

Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

This will:
1. Create a virtual environment in `venv/`
2. Activate it
3. Install all dependencies
4. Download WikiText-103
5. Clone nanoGPT

## Manual Setup

If you prefer to do it manually:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Using the Virtual Environment

Always activate the venv before working:

```bash
source venv/bin/activate
```

You'll see `(venv)` in your prompt.

To deactivate:
```bash
deactivate
```

## Remote Cluster

On the remote cluster, you can either:
1. Use the same venv approach (if you have write permissions)
2. Use conda environments (if available)
3. Use system Python with pip install

Update the SLURM scripts to activate your environment:
```bash
# Option 1: venv
source /path/to/Complexity_profiles/venv/bin/activate

# Option 2: conda
module load anaconda
conda activate mcp_env

# Option 3: system
pip install -r requirements.txt
```

## Verifying Installation

After setup, verify:
```bash
source venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import sklearn; print(f'sklearn: {sklearn.__version__}')"
```
