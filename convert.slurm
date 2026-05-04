#!/bin/bash
#SBATCH --job-name=convert_data
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4      # Use a CPU partition (change 'cpu' if AAU uses a different name for CPU nodes)
#SBATCH --cpus-per-task=32   # Grab 32 cores to rip through the files
#SBATCH --mem=64G            # Plenty of memory to hold the paths

set -e

# Go to the directory where the script is
cd /ceph/project/SW10-CausalLM/ciphertext-generation

# Add uv to path if necessary
export PATH="$HOME/.local/bin:$PATH"

# Force uv to create/use the .venv in your home directory
export UV_PROJECT_ENVIRONMENT="$HOME/.venv"

# Sync dependencies to ensure the .venv is completely up-to-date
echo "Syncing project dependencies to $UV_PROJECT_ENVIRONMENT..."
uv sync

echo "Starting conversion $@..."
uv run src/preprocess.py "$@"
echo "Finished completely!"