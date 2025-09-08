#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=script_fits       # Nombre del trabajo
#SBATCH --output=script_fits_%j.out  # Archivo de salida (stdout)
#SBATCH --error=script_fits_%j.err   # Archivo de errores (stderr)
#SBATCH --partition=general           # Partición a utilizar
#SBATCH --nodes=1                    # Número de nodos
#SBATCH --ntasks=1                   # Número de tareas
#SBATCH --cpus-per-task=4           # Número de CPUs por tarea
#SBATCH --mem=16G                   # Memoria por nodo
#SBATCH --time=24:00:00             # Tiempo máximo de ejecución (formato D-HH:MM:SS)
#SBATCH --mail-type=END,FAIL        # Notificaciones por correo electrónico al finalizar o fallar
#SBATCH --mail-user=elena.ramos@dipc.org  # Tu dirección de correo electrónico

# Cargar el entorno de Python (ajusta según sea necesario)
source /scratch/elena/elena_wcsim/build/env_wcsim.sh

python3 /scratch/elena/WCTE_2025_commissioning/2025_data/WCTE_BRB_Data_Analysis/script_fits_NEW.py
