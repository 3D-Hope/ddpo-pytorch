# SLURM Setup Guide for DDPO-PyTorch

This guide explains how to use the automated SLURM script to run DDPO training.

## Quick Start

1. **Set your Wandb API key** (optional, but recommended):
   ```bash
   export WANDB_API_KEY="your-wandb-api-key-here"
   ```

2. **Customize SLURM parameters** (if needed):
   Edit `run_slurm.sh` and modify the `#SBATCH` directives at the top:
   - `--job-name`: Job name
   - `--partition`: SLURM partition
   - `--gpus`: Number and type of GPUs
   - `--time`: Maximum runtime
   - `--nodelist`: Specific node (optional)

3. **Set project directory** (if not running from project root):
   ```bash
   export PROJECT_DIR="/path/to/ddpo-pytorch"
   ```

4. **Set conda directory** (if using custom location):
   ```bash
   export CONDA_DIR="/scratch/your-username/tools/miniforge"
   ```

5. **Submit the job**:
   ```bash
   sbatch run_slurm.sh
   ```

## What the Script Does

The script automatically:

1. **Sets up Miniforge** (if not already installed)
2. **Creates/activates conda environment** `ddpo-pytorch` with Python 3.11
3. **Installs dependencies**:
   - Upgrades pip
   - Installs compatible `huggingface_hub<0.26.0` (required for diffusers 0.17.1)
   - Runs `pip install -e .` to install the project
4. **Verifies critical packages** (torch, diffusers, accelerate, wandb, transformers)
5. **Logs into Wandb** (if API key provided)
6. **Runs training** using `accelerate launch scripts/train.py`

## Configuration Variables

You can customize these by setting environment variables before running:

- `WANDB_API_KEY`: Your Wandb API key (required for wandb logging)
- `WANDB_ENTITY`: Wandb entity/team (default: `078bct021-ashok-d`)
- `PROJECT_DIR`: Path to ddpo-pytorch directory (default: current directory)
- `CONDA_DIR`: Path to Miniforge installation (default: `/scratch/${USER}/tools/miniforge`)

## Example Usage

```bash
# Set your wandb API key
export WANDB_API_KEY="your-api-key-here"

# Optional: Set custom paths
export PROJECT_DIR="/home/your-username/codes/ddpo-pytorch"
export CONDA_DIR="/scratch/your-username/tools/miniforge"

# Submit job
sbatch run_slurm.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/ddpo_pytorch-<job-id>.out
```

## Customizing Training Configuration

To use a different config file, modify the last line in `run_slurm.sh`:

```bash
# Use default config
accelerate launch scripts/train.py --config config=config/base.py

# Use custom config
accelerate launch scripts/train.py --config config=config/your_config.py

# Override specific config parameters
accelerate launch scripts/train.py \
    --config config=config/base.py \
    --config.train.learning_rate=1e-4 \
    --config.sample.batch_size=2
```

## Troubleshooting

### Wandb Login Issues

If wandb login fails, the script will continue but wandb logging may not work. Make sure:
- `WANDB_API_KEY` is set correctly
- You have internet access from the compute node
- Your wandb account has proper permissions

### Conda Environment Issues

If the conda environment creation fails:
- Check disk space: `df -h /scratch/${USER}`
- Verify Miniforge installation: `ls -la $CONDA_DIR`
- Try removing and recreating: `conda env remove -n ddpo-pytorch`

### Import Errors

If you see import errors:
- Check that `huggingface_hub<0.26.0` is installed: `pip list | grep huggingface`
- Verify Python version: `python --version` (should be 3.11)
- Check installed packages: `pip list`

## Notes

- The script installs `huggingface_hub<0.26.0` automatically to avoid the `cached_download` import error
- Logs are saved to `logs/` directory
- The script uses `set -euo pipefail` for strict error handling
- All stages are logged with clear markers for easy debugging

