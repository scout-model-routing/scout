#!/bin/bash
# Download SCOUT dataset files from HuggingFace

set -e

echo "Downloading SCOUT dataset files..."
pip install -q huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Akashk1010/scout-routing-data', repo_type='dataset', local_dir='data/')"
echo "Done. Data files are in data/"
