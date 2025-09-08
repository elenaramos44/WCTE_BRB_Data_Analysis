#!/bin/bash
#SBATCH --job-name=save_pmt_wf_2runs
#SBATCH --output=logs/save_pmt_wf_%j.log
#SBATCH --error=logs/save_pmt_wf_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G

module purge
module load Python/3.7.4-GCCcore-8.3.0

# Directorio base con los ROOT files
DATADIR="/dipc/elena/WCTE_2025_commissioning/root_files/PMTs_calib"

# Directorios de salida para cada run
OUTDIR_2521="/scratch/elena/WCTE_2025_commissioning/pmt_arrays/run2521"
OUTDIR_2519="/scratch/elena/WCTE_2025_commissioning/pmt_arrays/run2519"

# Ruta al script Python (ajusta si es necesario)
SCRIPT="/scratch/elena/WCTE_2025_commissioning/2025_data/WCTE_BRB_Data_Analysis/save_pmt_waveforms.py"

echo "Processing run 2521 ..."
python $SCRIPT --run 2521 --outdir $OUTDIR_2521 --max_events 2000 --datadir $DATADIR

echo "Processing run 2519 ..."
python $SCRIPT --run 2519 --outdir $OUTDIR_2519 --max_events 2000 --datadir $DATADIR

echo "Done!"
