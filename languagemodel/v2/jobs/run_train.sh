#!/bin/bash
#SBATCH --job-name=spectra_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=40G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v2

python finetunemodel.py --data data/sample1_spectra_training_data.jsonl --epochs 1