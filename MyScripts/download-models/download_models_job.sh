#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=08:00:00
#PBS -N download_llama_models
#PBS -o /rds/general/user/yl9422/home/files
#PBS -e /rds/general/user/yl9422/home/files
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R
source ~/.bashrc
source /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/discord-notif/discord_notif.sh

cd /rds/general/user/yl9422/home/files/models

sh download_models.sh
