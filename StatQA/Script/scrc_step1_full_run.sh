#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=8:00:00
#PBS -N scrc_step1_full_run
#PBS -o /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/SCRC/outputs/step1_full
#PBS -e /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/SCRC/outputs/step1_full

set -e

cd /rds/general/user/yl9422/home/files/M4R-Elly/StatQA
mkdir -p SCRC/outputs/step1_full

echo "[i] Host: $(hostname)"
echo "[i] Start: $(date '+%Y-%m-%d %H:%M:%S')"

export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R
source ~/.bashrc
source /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/discord-notif/discord_notif.sh

/rds/general/user/yl9422/home/miniforge3/envs/M4R/bin/python SCRC/hidden_state_extractor.py \
  --batch_size 10 \
  --max_new_tokens 256 \
  --hidden_state_dir "SCRC/outputs/step1_full" \
  --origin_answer_dir "Model Answer/Origin Answer" \
  --model_name "llama3_8b"

echo "[i] End: $(date '+%Y-%m-%d %H:%M:%S')"
