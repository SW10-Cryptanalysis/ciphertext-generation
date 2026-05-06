#!/bin/bash
#SBATCH --job-name=monoalphabetic_conversion
#SBATCH --output=logs/monoalphabetic_conversion_%j.out   # Standard output/error log (%j expands to jobID)
#SBATCH --time=12:00:00                     # Max wall time limit (hours:minutes:seconds)
#SBATCH --nodes=1                           # We only need one node
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --cpus-per-task=2                   # Number of CPU cores to allocate
#SBATCH --mem=8G                            # Memory limit per node (8GB is plenty for our stream)
#SBATCH --partition=l4                      # Target partition

# Exit on any error
set -e

SPLIT=$1

if [ -z "$SPLIT" ]; then
    echo "Error: No split provided. Usage: sbatch $0 <SplitName> (e.g., Training or Validation)"
    exit 1
fi

if [ "$SPLIT" != "Training" ] && [ "$SPLIT" != "Validation" ]; then
    echo "Error: Invalid split name. Must be either Training or Validation."
    exit 1
fi

echo "Starting cipher truncation job..."
echo "Running on node: $SLURM_JOB_NODELIST"

# Create the logs and output directories if they don't exist
mkdir -p logs
mkdir -p ../Ciphers/Monoalphabetic/$SPLIT

# Navigate to the project root so uv can resolve the project environment
cd /ceph/project/SW10-CausalLM/ciphertext-generation

# Execute the pipeline directly through uv.
# uv automatically handles the environment isolation.
srun uv run src/convert_to_monoalphabetic.py \
    --directory ../Ciphers/$SPLIT \
    --output ../Ciphers/Monoalphabetic/$SPLIT

echo "Job completed successfully!"