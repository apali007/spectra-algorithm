#!/bin/bash
#SBATCH --job-name=spectra_train
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=100G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra

export HF_HOME=/disk/dragon-storage/homes/apali007/.cache/huggingface
export PYTHONUNBUFFERED=1

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4

# Run name ties together logs/, checkpoints/, and this job's SLURM ID
RUN_NAME="train_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/$RUN_NAME

# Ensure the staged SLURM logs get moved into the run folder no matter how the job exits
trap 'mv logs/_staging/${SLURM_JOB_ID}.log logs/$RUN_NAME/train.log 2>/dev/null; mv logs/_staging/${SLURM_JOB_ID}.err logs/$RUN_NAME/train.err 2>/dev/null' EXIT

python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/src/train.py \
  --epochs 1 \
  --output_dir /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/$RUN_NAME