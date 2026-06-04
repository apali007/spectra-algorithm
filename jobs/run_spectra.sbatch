#!/bin/bash
#SBATCH --job-name=spectra_job
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/logs/output_%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/logs/error_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=16G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra

cd ~/projects/spectra_algorithm
python scripts/bottom_up_spectra_algorithm.py