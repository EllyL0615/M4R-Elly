#!/bin/bash
#PBS -l select=1:ncpus=1:mem=12gb:ngpus=1
#PBS -l walltime=00:15:00
#PBS -N hs_gen_smoke
#PBS -o /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/SCRC/outputs/step1_smoke
#PBS -e /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/SCRC/outputs/step1_smoke

set -e

cd /rds/general/user/yl9422/home/files/M4R-Elly/StatQA
mkdir -p SCRC/outputs/step1_smoke

echo "[i] Host: $(hostname)"
echo "[i] Start: $(date '+%Y-%m-%d %H:%M:%S')"

export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R
source ~/.bashrc
source /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/discord-notif/discord_notif.sh

/rds/general/user/yl9422/home/miniforge3/envs/M4R/bin/python SCRC/1_hidden_state_gen.py \
  --smoke_rows_per_split 3 \
  --batch_size 10 \
  --max_new_tokens 256 \
  --hidden_state_dir "SCRC/outputs/step1_smoke" \
  --origin_answer_dir "Model Answer/Origin Answer" \
  --model_name "llama3_8b_smoke"

echo "[i] End: $(date '+%Y-%m-%d %H:%M:%S')"
