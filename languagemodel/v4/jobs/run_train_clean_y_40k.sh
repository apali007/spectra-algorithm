#!/bin/bash
#SBATCH --job-name=spectra_train_clean_y
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=100G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra
export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/spectra/lib:$LD_LIBRARY_PATH
export HF_HOME=/disk/dragon-storage/homes/apali007/.cache/huggingface
export PYTHONUNBUFFERED=1

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4

RUN_NAME="train_clean_y_40k_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/$RUN_NAME

trap 'mv logs/_staging/${SLURM_JOB_ID}.log logs/$RUN_NAME/train.log 2>/dev/null; mv logs/_staging/${SLURM_JOB_ID}.err logs/$RUN_NAME/train.err 2>/dev/null' EXIT

torchrun --nproc_per_node 2 \
  /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/src/train.py \
  --epochs 1 \
  --train_set_dir /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/clean_y_ion_only_full_mass_analyzer_aware_train.json \
  --val_set_dir /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/clean_y_ion_only_full_mass_analyzer_aware_val.json \
  --output_dir /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/$RUN_NAME