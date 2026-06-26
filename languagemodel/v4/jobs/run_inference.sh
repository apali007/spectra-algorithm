#!/bin/bash
#SBATCH --job-name=spectra_inference
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/inference_%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/inference_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=40G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra

export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/spectra/lib:$LD_LIBRARY_PATH

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4

# python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/inference-spectra-v4.py --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/reasoning_dataset_test.json --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/results-inference-spectra-v4.json

python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/inference-spectra-v4.py --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/overlap_10_reasoning_dataset_test.json --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/results-inference-spectra-v4-10-OVERLAP.json

# python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/inference-spectra-v4.py --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/overlap_20_reasoning_dataset_test.json --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/results-inference-spectra-v4-20-OVERLAP.json