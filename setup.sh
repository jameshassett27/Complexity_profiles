#!/bin/bash
# Setup script for MCP pilot project

echo "Setting up MCP pilot project..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download WikiText-103
echo "Downloading WikiText-103..."
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1', split='train'); load_dataset('wikitext', 'wikitext-103-v1', split='validation'); load_dataset('wikitext', 'wikitext-103-v1', split='test'); print('WikiText-103 downloaded')"

# Clone nanoGPT for GPT-2 training
echo "Cloning nanoGPT..."
git clone https://github.com/karpathy/nanoGPT.git external/nanoGPT

# Test data pipeline
echo "Testing data pipeline..."
python data/wikitext103.py

# Test DEM architecture
echo "Testing DEM architecture..."
python models/dem.py

# Test MCP mappings
echo "Testing MCP mappings..."
python mcp/mappings.py

# Test MCP metrics
echo "Testing MCP metrics..."
python mcp/metrics.py

# Test MCP pipeline
echo "Testing MCP pipeline..."
python mcp/pipeline.py

# Test pilot ablation
echo "Testing pilot ablation..."
python mcp/pilot_ablation.py

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review the project structure in README.md"
echo "2. Follow the pilot plan in PILOT_PLAN.md"
echo "3. Start with Day 1-2: Data pipeline (already tested)"
echo "4. Then implement DEM training (Day 2-3)"
