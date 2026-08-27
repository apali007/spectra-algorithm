#!/bin/bash
#SBATCH --job-name=spectra_inference
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/inference_%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/inference_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra

export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/spectra/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4

python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/prospect-inference-spectra.py \
--batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/30k_all_cases_prospect_reasoning_dataset_half2.json \
--output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_all_cases_prospect_results-inference-spectra-half2.json

# CUDA_VISIBLE_DEVICES=0 python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/prospect-inference-spectra.py \
# --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/all_clean_y_ion_case_prospect_reasoning_dataset_half1.json \
# --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/all_clean_y_ion_case_prospect_results-inference-spectra-half1.json &

# CUDA_VISIBLE_DEVICES=1 python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/prospect-inference-spectra.py \
# --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/all_clean_y_ion_case_prospect_reasoning_dataset_half2.json \
# --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/all_clean_y_ion_case_prospect_results-inference-spectra-half2.json &

# CUDA_VISIBLE_DEVICES=0 python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/prospect-inference-spectra.py \
# --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/30k_all_cases_prospect_reasoning_dataset_half1.json \
# --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_all_cases_prospect_results-inference-spectra-half1.json &

# CUDA_VISIBLE_DEVICES=1 python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/prospect-inference-spectra.py \
# --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/30k_all_cases_prospect_reasoning_dataset_half2.json \
# --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_all_cases_prospect_results-inference-spectra-half2.json &

# wait


# python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/prospect-inference-spectra.py --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_test.json --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/prospect_results-inference-spectra.json

# python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/inference-spectra-v4.py --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/reasoning_dataset_test.json --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/results-inference-spectra-v4-0-OVERLAP.json

# python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/inference-spectra-v4.py --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/overlap_10_reasoning_dataset_test.json --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/results-inference-spectra-v4-10-OVERLAP.json

# python /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/inference-spectra-v4.py --batch /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/overlap_20_reasoning_dataset_test.json --output /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/results-inference-spectra-v4-20-OVERLAP.json