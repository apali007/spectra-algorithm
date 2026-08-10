#!/bin/bash
#SBATCH --job-name=spectra_train
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/train_%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/train_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=40G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra

export HF_HOME=/disk/dragon-storage/homes/apali007/.cache/huggingface
export PYTHONUNBUFFERED=1

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4

python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/prospect-finetunemodel-spectra.py --epochs 1

# python finetunemodel-spectra.py --data /lclhome/apali007/projects/spectra_algorithm/output/reasoning_dataset_correct_instances_TUM_third_pool_1_01_01_annotation.json --epochs 1

# python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/finetunemodel-spectra-v4.py --epochs 1

# merging LORA weights and full weights of QWEN instruct model
# python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/merge_only.py


