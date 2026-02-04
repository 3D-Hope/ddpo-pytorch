#!/bin/bash
#SBATCH --job-name=test_geometric
#SBATCH --partition=batch
#SBATCH --gpus=h200:1
#SBATCH --nodelist=sof1-h200-2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Better error reporting including line number
trap 'ERR_CODE=$?; echo "❌ Error on line ${LINENO:-?}. Exit code: $ERR_CODE" >&2; exit $ERR_CODE' ERR
trap 'echo "🛑 Job interrupted"; exit 130' INT

# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================
# Set your wandb API key here or pass via environment variable
WANDB_ENTITY="${WANDB_ENTITY:-078bct021-ashok-d}"

# Project directory (adjust to your actual path)
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

# Conda installation directory (adjust to your scratch space)
USER="pramish_paudel"
CONDA_DIR="${CONDA_DIR:-/scratch/${USER}/tools/miniforge}"

# Conda environment name
CONDA_ENV_NAME="ddpo-pytorch"

# Python version
DESIRED_PY="3.11"

# ============================================================================
# Basic setup / logging
# ============================================================================
mkdir -p logs
echo "──────────────────────────────────────────────────────────────────────────────"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Project directory: $PROJECT_DIR"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

echo "System information:"
free -h || true
df -h /scratch/${USER} 2>/dev/null || df -h . || true
echo ""

# ============================================================================
# STAGE 1: Miniforge/Conda setup
# ============================================================================
echo "STAGE 1: Setting up Miniforge (if missing)..."
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniforge to $CONDA_DIR..."
    mkdir -p "$(dirname "$CONDA_DIR")"
    cd "$(dirname "$CONDA_DIR")"
    MINIFORGE_SH="miniforge_installer.sh"
    wget -q --show-progress "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O "$MINIFORGE_SH" || {
        echo "❌ Failed to download Miniforge installer"; exit 1
    }
    bash "$MINIFORGE_SH" -b -p "$CONDA_DIR" || { echo "❌ Failed to install Miniforge"; exit 1; }
    rm -f "$MINIFORGE_SH"
    echo "✅ Miniforge installed at $CONDA_DIR"
else
    echo "✅ Miniforge already exists at $CONDA_DIR"
fi

# Source conda hooks reliably
echo "Sourcing conda..."
# shellcheck source=/dev/null
if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
else
    echo "❌ Expected conda.sh not found at $CONDA_DIR/etc/profile.d/conda.sh"; exit 1
fi
# Ensure conda command available in this shell
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)" || true

echo ""

# ============================================================================
# STAGE 2: Create and activate conda env
# ============================================================================
echo "STAGE 2: Creating/activating conda env '$CONDA_ENV_NAME' with python=$DESIRED_PY..."

# Create if missing
if ! "$CONDA_DIR/bin/conda" env list | awk '{print $1}' | grep -xq "$CONDA_ENV_NAME"; then
    echo "Creating conda environment: $CONDA_ENV_NAME (python=$DESIRED_PY)"
    "$CONDA_DIR/bin/conda" create -n "$CONDA_ENV_NAME" python="$DESIRED_PY" -y || {
        echo "❌ Failed to create conda env"; exit 1
    }
else
    echo "✅ Conda env $CONDA_ENV_NAME already present"
fi

# Activate environment
echo "Activating conda environment: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME" || { echo "❌ Failed to activate conda env"; exit 1; }

# Ensure conda python is first in PATH
export PATH="${CONDA_PREFIX:-$CONDA_DIR/envs/$CONDA_ENV_NAME}/bin:$PATH"
hash -r || true

echo "Environment verification:"
echo "  CONDA_PREFIX: ${CONDA_PREFIX:-N/A}"
echo "  Active conda environment: ${CONDA_DEFAULT_ENV:-N/A}"
echo "  Python path: $(which python)"
echo "  Python version: $(python --version 2>&1)"
echo "  Pip path: $(which pip)"
echo ""

# ============================================================================
# STAGE 3: Navigate to project directory and install dependencies
# ============================================================================
echo "STAGE 3: Installing project dependencies..."
cd "$PROJECT_DIR" || {
    echo "❌ Failed to change to project directory $PROJECT_DIR"; exit 1
}
echo "Current directory: $(pwd)"
echo ""

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip --quiet || { echo "❌ Failed to upgrade pip"; exit 1; }

# Install huggingface_hub compatible version first (required for diffusers 0.17.1)
echo "Installing compatible huggingface_hub version..."
pip install "huggingface_hub<0.26.0" --quiet || {
    echo "❌ Failed to install huggingface_hub"; exit 1
}

# Install project in editable mode
echo "Installing project in editable mode (pip install -e .)..."
pip install -e . || {
    echo "❌ Failed to install project"; exit 1
}

pip install bitsandbytes

echo "✅ Dependencies installed successfully"
echo ""

# ============================================================================
# STAGE 4: Verify critical packages
# ============================================================================
echo "STAGE 4: Verifying critical packages..."
python - <<'PYTEST' || { echo "❌ Required python imports failed"; exit 1; }
try:
    import importlib, sys
    modnames = ["torch", "diffusers", "accelerate", "wandb", "transformers"]
    missing = []
    for m in modnames:
        try:
            importlib.import_module(m)
        except Exception as e:
            missing.append((m, str(e)))
    if missing:
        print("MISSING:", missing)
        sys.exit(2)
    else:
        print("✅ All checks passed:", [importlib.import_module(m).__name__ for m in modnames])
except Exception as e:
    print("Import-time error:", str(e))
    raise
PYTEST

echo ""

# Set wandb entity
export WANDB_ENTITY="$WANDB_ENTITY"
echo "Wandb entity: $WANDB_ENTITY"
echo ""

# ============================================================================
# STAGE 6: GPU check
# ============================================================================
echo "STAGE 6: GPU check (nvidia-smi):"
nvidia-smi || echo "⚠️  nvidia-smi failed or not present on this node"
echo ""

# ============================================================================
# STAGE 7: Run training
# ============================================================================
echo "✅ All dependencies installed and configured"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "STAGE 7: Starting DDPO training (TESTING MODE)..."
echo "Training started at: $(date)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

export PYTHONUNBUFFERED=1
RUN_NAME="test_geometric_debug"

# ============================================================================
# STAGE 6.5: Verify GPU allocation
# ============================================================================
echo "STAGE 6.5: Verifying GPU allocation..."
echo "════════════════════════════════════════════════════════════════════════"
echo "SLURM GPU Allocation:"
echo "  SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-N/A}"
echo "  SLURM_GPUS: ${SLURM_GPUS:-N/A}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo ""

echo "PyTorch GPU Detection:"
python - <<'PYGPU'
import torch
import os

print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA device count: {torch.cuda.device_count()}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

if torch.cuda.is_available():
    print(f"\n  Detected GPUs:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        print(f"    GPU {i}: {props.name} ({mem_gb:.1f} GB)")
else:
    print("  ❌ No CUDA devices found!")
PYGPU

echo "════════════════════════════════════════════════════════════════════════"
echo ""


# Run accelerate launch
# NOTE: Overriding num_steps to 50 as requested
MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))

# Reduced num_processes for testing (1 GPU)
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --main_process_port=${MASTER_PORT} \
    scripts/train.py \
    --config=config/geometric.py \
    --config.run_name=$RUN_NAME \
    --config.sample.num_steps=10 \
    --config.incremental_training=False \
    --config.num_epochs=1 \
    --config.sample.batch_size=2




# ============================================================================
# Final status
# ============================================================================
EXIT_CODE=$?
echo ""
echo "════════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully at: $(date)"
    exit 0
else
    echo "❌ Training failed with exit code $EXIT_CODE at: $(date)"
    exit $EXIT_CODE
fi
echo "════════════════════════════════════════════════════════════════════════"
