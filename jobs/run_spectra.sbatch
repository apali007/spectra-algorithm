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
export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/spectra/lib:$LD_LIBRARY_PATH

cd ~/projects/spectra_algorithm
echo "Running on node: $(hostname)"
python scripts/practice.py