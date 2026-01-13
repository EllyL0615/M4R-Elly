#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=08:00:00
#PBS -N llama2_7b_job
#PBS -o /rds/general/user/yl9422/home/files
#PBS -e /rds/general/user/yl9422/home/files
#PBS -M yl9422@ic.ac.uk
#PBS -m abe

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R
source ~/.bashrc
source /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/discord-notif/discord_notif.sh

cd /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/Script

sh llama_exp.sh