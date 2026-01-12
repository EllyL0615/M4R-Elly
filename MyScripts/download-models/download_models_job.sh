#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=08:00:00
#PBS -N download_llama_models
#PBS -o /rds/general/user/yl9422/home/files
#PBS -e /rds/general/user/yl9422/home/files
#PBS -M yl9422@ic.ac.uk
#PBS -m abe

cd /rds/general/user/yl9422/home/files/models

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R

./download_models.sh
