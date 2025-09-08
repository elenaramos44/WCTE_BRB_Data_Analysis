#!/bin/bash
#SBATCH --job-name=test_awkward
#SBATCH --output=logs/test_awkward_%j.log
#SBATCH --error=logs/test_awkward_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=general

# Cargar entorno WCSim que contiene awkward y demás módulos
source /scratch/elena/elena_wcsim/build/env_wcsim.sh
source /scratch/elena/awkward_env/bin/activate
# Aquí ejecutas el python que quieres, por ejemplo:
python -c "import awkward as ak; print(ak.__version__)"

