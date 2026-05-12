#!/bin/bash

# Exit on any error
set -e

# 1. Navigate to your mounted UCloud workspace
cd /work

# 2. Clone the repository if it doesn't exist yet
if [ ! -d "ciphertext-generation" ]; then
    echo "Cloning repository..."
    git clone https://github.com/SW10-Cryptanalysis/ciphertext-generation.git
fi

# 3. Enter the project directory
cd ciphertext-generation
mkdir -p logs

# 4. Hardcode your target path here
TARGET_PATH="Monoalphabetic"

echo "Target folder : $TARGET_PATH"

# Sync dependencies to ensure the .venv is up-to-date
echo "Syncing project dependencies..."
uv sync

# 5. Run the python script writing directly to the target path
echo "Starting conversion..."
uv run src/preprocess.py --folder "$TARGET_PATH"

echo "Finished completely!"