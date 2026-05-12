#!/bin/bash

# Exit on any error
set -e

# Navigate to your mounted workspace
cd /work

# 2. Clone the repository and specific branch if it doesn't exist yet
if [ ! -d "ciphertext-generation" ]; then
    echo "Cloning repository..."
    git clone https://github.com/SW10-Cryptanalysis/ciphertext-generation.git
fi

cd ciphertext-generation
mkdir -p logs

# Hardcode the splits you want to process
SPLITS=("Training" "Validation")

for SPLIT in "${SPLITS[@]}"; do
    echo "Starting cipher truncation job for $SPLIT..."

    # Create the output directories if they don't exist
    mkdir -p ../Ciphers/Monoalphabetic/$SPLIT

    # Run the python script
    uv run src/convert_to_monoalphabetic.py \
        --directory ../Ciphers/$SPLIT \
        --output ../Ciphers/Monoalphabetic/$SPLIT

    echo "Job for $SPLIT completed successfully!"
    echo "----------------------------------------"
done

echo "All tasks finished!"