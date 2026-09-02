#!/bin/bash
#SBATCH --job-name=spectra_job
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/output_%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --nodes=1

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra
export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/spectra/lib:$LD_LIBRARY_PATH

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4
echo "Running on node: $(hostname)"

python src/reasoning_corpus.py

# python eval/precursor_mass_gap.py

# python archive/prospect_practice.py