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

# python finetunemodel-spectra.py --data /lclhome/apali007/projects/spectra_algorithm/output/reasoning_dataset_correct_instances_TUM_third_pool_1_01_01_annotation.json --epochs 1

python finetunemodel-spectra.py --data /lclhome/apali007/projects/spectra_algorithm/output/reasoning_dataset_6942_positive.json --epochs 1
