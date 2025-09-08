#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=save_pmt_wf
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/save_pmt_wf_%j.log
#SBATCH --error=logs/save_pmt_wf_%j.err




source /scratch/elena/elena_wcsim/build/env_wcsim.sh
export PYTHONPATH=$HOME/.local/lib/python3.7/site-packages:$PYTHONPATH

echo "Using python: $(which python)"
python --version
python -c "import awkward as ak; print('awkward version:', ak.__version__)"


cd /scratch/elena/WCTE_2025_commissioning/2025_data/WCTE_BRB_Data_Analysis
python save_pmt_waveforms.py \
    --run 2519 \
    --outdir /scratch/elena/WCTE_2025_commissioning/pmt_arrays/run2519 \
    --max_events 2000

