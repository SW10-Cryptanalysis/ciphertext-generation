#!/bin/bash
#SBATCH --job-name=convert_data
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4      # Use a CPU partition (change 'cpu' if AAU uses a different name for CPU nodes)
#SBATCH --cpus-per-task=32   # Grab 32 cores to rip through the files
#SBATCH --mem=64G            # Plenty of memory to hold the paths

set -e

OUTPUT_PATH=""
ARGS=("$@")

for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--output" || "${ARGS[i]}" == "-o" ]]; then
        OUTPUT_PATH="${ARGS[i+1]}"
        break
    fi
done

if [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: Could not find --output or -o argument."
    exit 1
fi

FOLDER_NAME=$(basename "$OUTPUT_PATH")
CEPH_PARENT_DIR=$(dirname "$OUTPUT_PATH")
LOCAL_TMP_DIR="/tmp/$SLURM_JOB_ID/$FOLDER_NAME"

echo "Detected target folder : $FOLDER_NAME"
echo "Routing processing to  : $LOCAL_TMP_DIR"

mkdir -p "/tmp/$SLURM_JOB_ID"

# Go to the directory where the script is
cd /ceph/project/SW10-CausalLM/ciphertext-generation

# Add uv to path if necessary
export PATH="$HOME/.local/bin:$PATH"

# Force uv to create/use the .venv in your home directory
export UV_PROJECT_ENVIRONMENT="$HOME/.venv"

# Sync dependencies to ensure the .venv is completely up-to-date
echo "Syncing project dependencies to $UV_PROJECT_ENVIRONMENT..."
uv sync

NEW_ARGS=()
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--output" || "${ARGS[i]}" == "-o" ]]; then
        NEW_ARGS+=("${ARGS[i]}" "$LOCAL_TMP_DIR")
        ((i++)) # Skip the original Ceph path
    else
        NEW_ARGS+=("${ARGS[i]}")
    fi
done

echo "Starting conversion..."
uv run src/preprocess.py "${NEW_ARGS[@]}"
echo "Local processing finished!"

echo "Transferring data from local scratch to Ceph..."
mkdir -p "$CEPH_PARENT_DIR"
cp -r "$LOCAL_TMP_DIR" "$CEPH_PARENT_DIR/"

rm -rf "/tmp/$SLURM_JOB_ID"

echo "Finished completely!"