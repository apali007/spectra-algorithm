#!/bin/bash
#SBATCH --job-name=spectra_job
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/logs/output_%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/logs/error_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=16G

cd ~/projects/spectra_algorithm
python3 scripts/bottom_up_MS_generate_spectra_description.py
