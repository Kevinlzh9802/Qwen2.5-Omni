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
#   sbatch job_scripts/inference_batch_test.sh -set test_run -model 7B -prompt plain
#   sbatch job_scripts/inference_batch_test.sh -set test_run -model 3B -prompt intention

set -euo pipefail

model_size=7B
set_name=""
prompt_choice=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        -set|--set)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            set_name="$2"
            shift 2
            ;;
        -model|--model)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            model_size="$2"
            shift 2
            ;;
        -prompt|--prompt)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            prompt_choice="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: sbatch job_scripts/inference_batch_test.sh -set <dataset_folder> [-model 7B|3B] -prompt <prompt_choice>" >&2
            exit 0
            ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            exit 1
            ;;
        *)
            echo "[ERROR] Unexpected positional argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$set_name" ]; then
    echo "Usage: sbatch job_scripts/inference_batch_test.sh -set <dataset_folder> [-model 7B|3B] -prompt <prompt_choice>" >&2
    exit 1
fi

if [ -z "$prompt_choice" ]; then
    echo "Usage: sbatch job_scripts/inference_batch_test.sh -set <dataset_folder> [-model 7B|3B] -prompt <prompt_choice>" >&2
    exit 1
fi

case "$model_size" in
    7B|3B)
        ;;
    *)
        echo "[ERROR] Invalid model size: $model_size (expected 7B or 3B)" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir=/home/zli33/projects/Qwen2.5-Omni
sif_file=/scratch/zli33/apptainers/qwen2.5-omni-inference.sif
hf_cache_host=/scratch/zli33/.cache/huggingface
data_root_host=/scratch/zli33/data
model_root_host=/scratch/zli33/models
gestalt_root=/scratch/zli33/data/gestalt_bench

# Batch-specific paths
data_parent="${gestalt_root}/${set_name}"
output_dir="${gestalt_root}/results/qwen2.5/${set_name}/${model_size}_${prompt_choice}"
output_json="$output_dir/results.json"
model_path="/scratch/zli33/models/Qwen2.5-Omni-${model_size}"
prompt_first="$project_dir/prompts/${prompt_choice}_1.txt"
prompt_after="$project_dir/prompts/${prompt_choice}_after.txt"

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

if [ ! -d "$data_parent" ]; then
    echo "[ERROR] Dataset folder not found: $data_parent" >&2
    exit 1
fi

if [ ! -d "$model_path" ]; then
    echo "[ERROR] Model path not found: $model_path" >&2
    exit 1
fi

if [ ! -f "$prompt_first" ]; then
    echo "[ERROR] First-turn prompt not found: $prompt_first" >&2
    exit 1
fi

if [ ! -f "$prompt_after" ]; then
    echo "[ERROR] Follow-up prompt not found: $prompt_after" >&2
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
echo "[INFO] set_name    = $set_name"
echo "[INFO] data_parent = $data_parent"
echo "[INFO] model_size  = $model_size"
echo "[INFO] model_path  = $model_path"
echo "[INFO] prompt_choice = $prompt_choice"
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
        --output "$output_json" \
        --prompt-choice "$prompt_choice"

echo ""
echo "[INFO] Batch inference completed. Results saved to $output_json"
