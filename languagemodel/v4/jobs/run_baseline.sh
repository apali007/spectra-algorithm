#!/bin/bash
#SBATCH --job-name=spectra_job
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --nodes=1

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate casanovo_env
export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/casanovo_env/lib:$LD_LIBRARY_PATH

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4
echo "Running on node: $(hostname)"

RUN_NAME="baseline_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/$RUN_NAME

trap 'mv /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/${SLURM_JOB_ID}.log /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/$RUN_NAME/output.log 2>/dev/null; mv /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/${SLURM_JOB_ID}.err /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/$RUN_NAME/error.err 2>/dev/null' EXIT

cd baselines/casanovo/results
casanovo sequence /disk/dragon-storage/homes/apali007/projects/spectra_algorithm/languagemodel/v4/baselines/casanovo/input_data/TUM_third_pool/clean_y_ion_only_full_mass_analyzer_aware/test.mgf --evaluate