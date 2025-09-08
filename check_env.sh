#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --output=logs/check_env_%j.log

cat /dipc/elena/myenv/bin/activate

