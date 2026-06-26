#!/bin/bash
#SBATCH --job-name=spectra_job
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/logs/output_%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/logs/error_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=compute
#SBATCH --nodes=1

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra

cd ~/projects/spectra_algorithm
# python scripts/bottom_up_spectra_algorithm_reasoning_dataset_v2.py

# python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/prospect-reasoning-corpus.py

python scripts/practice.py