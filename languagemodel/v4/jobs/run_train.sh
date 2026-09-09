#!/bin/bash
#SBATCH --job-name=spectra_train
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=100G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra
export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/spectra/lib:$LD_LIBRARY_PATH
export HF_HOME=/disk/dragon-storage/homes/apali007/.cache/huggingface
export PYTHONUNBUFFERED=1

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4

RUN_LABEL="${1:-train}"
TRAIN_SET_DIR="$2"
VAL_SET_DIR="$3"
EARLY_STOPPING="${4:-}"

# Rename the job (visible in squeue) now that we know RUN_LABEL --
# #SBATCH directives are fixed at submission time and can't use variables,
# so this is done via scontrol once the job is actually running instead.
scontrol update JobId=$SLURM_JOB_ID JobName="spectra_train_${RUN_LABEL}"

RUN_NAME="train_${RUN_LABEL}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/$RUN_NAME

trap 'mv logs/_staging/${SLURM_JOB_ID}.log logs/$RUN_NAME/train.log 2>/dev/null; mv logs/_staging/${SLURM_JOB_ID}.err logs/$RUN_NAME/train.err 2>/dev/null' EXIT

EXTRA_ARGS=()
if [ -n "$TRAIN_SET_DIR" ]; then
  EXTRA_ARGS+=(--train_set_dir "$TRAIN_SET_DIR")
fi
if [ -n "$VAL_SET_DIR" ]; then
  EXTRA_ARGS+=(--val_set_dir "$VAL_SET_DIR")
fi
if [ "$EARLY_STOPPING" = "early_stopping" ]; then
  EXTRA_ARGS+=(--early_stopping)
fi

torchrun --nproc_per_node 2 \
  /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/src/train.py \
  --epochs 1 \
  --output_dir /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/$RUN_NAME \
  "${EXTRA_ARGS[@]}"