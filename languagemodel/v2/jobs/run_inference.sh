#!/bin/bash
#SBATCH --job-name=spectra_inference
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v2/logs/inference_%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v2/logs/inference_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=40G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra

export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/spectra/lib:$LD_LIBRARY_PATH

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v2

python inference.py --checkpoint spectra-chat-checkpoints/final --batch data/sample1_spectra_training_data.jsonl --output results.json

