#!/bin/bash
#PBS -l select=1:ncpus=1:mem=12gb:ngpus=1
#PBS -l walltime=00:15:00
#PBS -N hs_gen_smoke
#PBS -o /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/SCRC/data/data-smoke
#PBS -e /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/SCRC/data/data-smoke

set -e

# Which LLaMA model to run. Override at submit time, e.g.:
#   qsub -v MODEL_TYPE=3_2_1b Script/scrc_step1_hs_gen_smoke.sh
# Choices: 3_1_8b | 3_2_1b | 3_2_3b
MODEL_TYPE="${MODEL_TYPE:-3_1_8b}"

cd /rds/general/user/yl9422/home/files/M4R-Elly/StatQA
mkdir -p SCRC/data/data-smoke

echo "[i] Host: $(hostname)"
echo "[i] Model type: ${MODEL_TYPE}"
echo "[i] Start: $(date '+%Y-%m-%d %H:%M:%S')"

export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R_llama3x
source ~/.bashrc
source /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/discord-notif/discord_notif.sh

/rds/general/user/yl9422/home/miniforge3/envs/M4R_llama3x/bin/python SCRC/1_hidden_state_gen.py \
  --model_type "$MODEL_TYPE" \
  --smoke_rows_per_split 3 \
  --batch_size 10 \
  --max_new_tokens 256 \
  --hidden_state_dir "SCRC/data/data-smoke" \
  --origin_answer_dir "SCRC/data/data-smoke"

echo "[i] End: $(date '+%Y-%m-%d %H:%M:%S')"
