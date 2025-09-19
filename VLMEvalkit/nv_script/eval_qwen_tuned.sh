#!/bin/bash
# CRITICAL: Override any distributed environment for single-process execution
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT
# Also clear SLURM variables
unset SLURM_PROCID
unset SLURM_LOCALID
unset SLURM_NTASKS
unset SLURM_NPROCS

# Initialize CONFIG_FILE as empty
CONFIG_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 --config CONFIG_FILE"
            exit 1
            ;;
    esac
done

# Check if CONFIG_FILE is provided
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config CONFIG_FILE is required"
    echo "Usage: $0 --config CONFIG_FILE"
    exit 1
fi

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Using $NUM_GPUS GPU(s)"

# Add user site-packages to Python path
export PYTHONPATH="/home/ymingli/.local/lib/python3.10/site-packages:$PYTHONPATH"
# Source conda and activate environment
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate allanb
export HF_HUB_CACHE="/lustre/fsw/portfolios/nvr/users/ymingli/cache/huggingface/hub"

cd "/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/github/all_angles_bench/VLMEvalkit"

torchrun --nproc-per-node=$NUM_GPUS run.py --config "$CONFIG_FILE" --verbose --work-dir "/lustre/fsw/portfolios/nvr/users/ymingli/projects/playground/github/all_angles_bench/VLMEvalkit"