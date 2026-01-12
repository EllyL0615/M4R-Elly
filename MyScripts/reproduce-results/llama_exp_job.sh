#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=04:00:00
#PBS -N llama2_7b_job
#PBS -o /rds/general/user/yl9422/home/files
#PBS -e /rds/general/user/yl9422/home/files

cd /rds/general/user/yl9422/home/files/M4R-Elly/StatQA

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R

sh Script/llama_exp.sh