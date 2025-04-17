#!/bin/bash

# === CONFIG ===
echo "Configuring paths..."
REQUIRED_ENV="/eos/home-d/dcostasr/conda_envs/wcte"
RUNS_DIR="/eos/experiment/wcte/data/2025_commissioning/offline_data"
PYTHON_SCRIPT="/eos/home-d/dcostasr/SWAN_projects/2025_data/wcte/create_df.py"

echo "Checking you have the correct conda env..."
# === CHECK CONDA ENV ===
if [[ "$CONDA_DEFAULT_ENV" != "$REQUIRED_ENV" ]]; then
    echo "ERROR: This script requires that the conda env '$REQUIRED_ENV' is activated."
    echo "Current env: '$CONDA_DEFAULT_ENV'"
    exit 1
fi

# # === OBTAIN RUN LIST ===
# cd "$RUNS_DIR" || { echo "Cannot acces run directory $RUNS_DIR"; exit 1; }

# # === COLLECT AND SORT RUN DIRECTORIES (numeric names only) ===
# run_numbers=()
# for d in */; do
#     dir="${d%/}"  # remove trailing slash
#     if [[ "$dir" =~ ^[0-9]+$ ]]; then
#         run_numbers+=("$dir")
#     fi
# done

# # Sort run numbers numerically
# IFS=$'\n' sorted_runs=($(sort -n <<<"${run_numbers[*]}"))
# unset IFS

# === PROCESS EACH RUN ONE BY ONE ===
#for run in "${sorted_runs[@]}"; do
#    echo "Processing run $run..."
#    python "$PYTHON_SCRIPT" "$run"
#done

echo "Executing Python Script..."
python "$PYTHON_SCRIPT" 1570