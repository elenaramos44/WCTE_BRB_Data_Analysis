#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=test_env
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --output=logs/test_env_%j.log
#SBATCH --error=logs/test_env_%j.err

source /scratch/elena/myenv/bin/activate
which python
python -c "import awkward; print('awkward version:',
awkward.__version__)"

