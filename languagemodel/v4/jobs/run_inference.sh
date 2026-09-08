#!/bin/bash
#SBATCH --job-name=spectra_inference
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra

export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/spectra/lib:$LD_LIBRARY_PATH
export HF_HOME=/disk/dragon-storage/homes/apali007/.cache/huggingface
export PYTHONUNBUFFERED=1

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4

RUN_LABEL="${1:-test}"
CHECKPOINT_DIR="$2"
TEST_SET_DIR="$3"

# Fail fast with a clear message before any GPU work starts, rather than
# letting inference.py's own path checks catch it after both models load.
if [ -z "$CHECKPOINT_DIR" ] || [ -z "$TEST_SET_DIR" ]; then
  echo "Usage: sbatch run_inference.sh <run_label> <checkpoint_dir> <test_set_dir>"
  echo "Missing one or more required arguments."
  exit 1
fi

scontrol update JobId=$SLURM_JOB_ID JobName="spectra_inference_${RUN_LABEL}"

RUN_NAME="inference_${RUN_LABEL}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="output/$RUN_NAME"

mkdir -p logs/$RUN_NAME
mkdir -p "$OUTPUT_DIR"

trap 'mv logs/_staging/${SLURM_JOB_ID}.log logs/$RUN_NAME/inference.log 2>/dev/null; mv logs/_staging/${SLURM_JOB_ID}.err logs/$RUN_NAME/inference.err 2>/dev/null' EXIT


# Both GPUs read the SAME test set file; --shard_index/--num_shards inside
# inference.py splits it into disjoint halves, so no pre-split file is
# needed and there's no output collision.
CUDA_VISIBLE_DEVICES=0 python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/src/inference.py \
  --checkpoint "$CHECKPOINT_DIR" \
  --batch "$TEST_SET_DIR" \
  --output "$OUTPUT_DIR/shard0_test_set_results.json" \
  --shard_index 0 --num_shards 2 \
  &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/src/inference.py \
  --checkpoint "$CHECKPOINT_DIR" \
  --batch "$TEST_SET_DIR" \
  --output "$OUTPUT_DIR/shard1_test_set_results.json" \
  --shard_index 1 --num_shards 2 \
  &
PID2=$!

wait $PID1; STATUS1=$?
wait $PID2; STATUS2=$?

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
  echo "Inference results:"

  if [ $STATUS1 -eq 0 ]; then
    echo "  GPU 0 | shard 0: SUCCESS"
  else
    echo "  GPU 0 | shard 0: FAILED (exit code: $STATUS1)"
  fi

  if [ $STATUS2 -eq 0 ]; then
    echo "  GPU 1 | shard 1: SUCCESS"
  else
    echo "  GPU 1 | shard 1: FAILED (exit code: $STATUS2)"
  fi

  exit 1
fi

echo "Inference results:"
echo "  GPU 0 | shard 0: SUCCESS"
echo "  GPU 1 | shard 1: SUCCESS"