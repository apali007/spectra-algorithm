#!/bin/bash
#SBATCH --job-name=spectra_job
#SBATCH --output=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.log
#SBATCH --error=/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/logs/_staging/%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --nodes=1

source /lclhome/apali007/miniconda3/etc/profile.d/conda.sh
conda activate spectra
export LD_LIBRARY_PATH=/lclhome/apali007/miniconda3/envs/spectra/lib:$LD_LIBRARY_PATH

cd /lclhome/apali007/projects/spectra_algorithm/languagemodel/v4
echo "Running on node: $(hostname)"

RUN_NAME="reasoning_corpus_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/$RUN_NAME

trap 'mv logs/_staging/${SLURM_JOB_ID}.log logs/$RUN_NAME/output.log 2>/dev/null; mv logs/_staging/${SLURM_JOB_ID}.err logs/$RUN_NAME/error.err 2>/dev/null' EXIT

python src/reasoning_corpus.py

# python eval/precursor_mass_gap.py

# python archive/prospect_practice.py