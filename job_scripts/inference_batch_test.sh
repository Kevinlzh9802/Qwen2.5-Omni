#!/bin/bash
#SBATCH --job-name="qwen2.5-omni_batch_infer"
#SBATCH --partition=gpu-a100
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/qwen2.5-omni/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/qwen2.5-omni/slurm_%j.err

# Batch Qwen2.5-Omni inference via Apptainer.
#
# Submit from the project folder:
#   sbatch job_scripts/inference_batch_test.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir=/home/zli33/projects/Qwen2.5-Omni
sif_file=/scratch/zli33/apptainers/qwen2.5-omni-inference.sif
hf_cache_host=/scratch/zli33/.cache/huggingface
data_root_host=/scratch/zli33/data
model_root_host=/scratch/zli33/models

# Batch-specific paths
data_parent=/scratch/zli33/data/gestalt_bench/test_run_part
output_dir=/scratch/zli33/data/gestalt_bench/results/qwen2.5
output_json="$output_dir/results.json"
model_path=/scratch/zli33/models/Qwen2.5-Omni-7B

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2
    echo "  Build/copy the SIF first." >&2
    exit 1
fi

if [ ! -f "$project_dir/batch_infer.py" ]; then
    echo "[ERROR] batch_infer.py not found in: $project_dir" >&2
    exit 1
fi

if [ ! -f "$project_dir/prompt.txt" ]; then
    echo "[ERROR] prompt.txt not found in: $project_dir" >&2
    exit 1
fi

# Ensure output and cache directories exist
mkdir -p /scratch/zli33/slurm_outputs/qwen2.5-omni
mkdir -p "$hf_cache_host"
mkdir -p "$output_dir"

# ---------------------------------------------------------------------------
# Run batch inference
# ---------------------------------------------------------------------------
echo "[INFO] sif_file    = $sif_file"
echo "[INFO] project_dir = $project_dir"
echo "[INFO] hf_cache    = $hf_cache_host"
echo "[INFO] data_parent = $data_parent"
echo "[INFO] model_path  = $model_path"
echo "[INFO] output_json = $output_json"
echo ""

apptainer exec --nv \
    --bind "$project_dir":/workspace \
    --bind "$hf_cache_host":/opt/huggingface \
    --bind "$data_root_host":"$data_root_host" \
    --bind "$model_root_host":"$model_root_host" \
    --env HF_HOME=/opt/huggingface \
    --env TRANSFORMERS_CACHE=/opt/huggingface \
    --pwd /workspace \
    "$sif_file" \
    python batch_infer.py \
        --model "$model_path" \
        --data-root "$data_parent" \
        --output "$output_json"

echo ""
echo "[INFO] Batch inference completed. Results saved to $output_json"
