#!/bin/bash
set -eo pipefail

cd /work

if [ ! -d ciphertext-generation ]; then
    echo "Cloning repository..."
    git clone https://github.com/SW10-Cryptanalysis/ciphertext-generation.git
    cd ciphertext-generation
else
    echo "Updating repository..."
    cd ciphertext-generation
    git pull
fi

mkdir -p logs

export UV_CACHE_DIR="/work/.uv_cache"

echo "Starting cipher metadata generation job..."

uv run src/generate_stats.py --directory ../Ciphers

echo "Training Job finished at $(date)"****