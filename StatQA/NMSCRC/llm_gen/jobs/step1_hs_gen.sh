#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=6:00:00
#PBS -N nmscrc_hs_gen
#PBS -o /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/NMSCRC/data/data-full
#PBS -e /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/NMSCRC/data/data-full

set -e

# Step 1 worker: extract hidden states + method answers for ONE model.
# Which LLaMA model to run. Override at submit time, e.g.:
#   qsub -v MODEL_TYPE=3_2_1b llm_gen/jobs/step1_hs_gen.sh
# Choices: 3_1_8b | 3_2_1b | 3_2_3b
MODEL_TYPE="${MODEL_TYPE:-3_1_8b}"

NMSCRC_ROOT=/rds/general/user/yl9422/home/files/M4R-Elly/StatQA/NMSCRC
cd "$NMSCRC_ROOT"
mkdir -p data/data-full

echo "[i] Host: $(hostname)"
echo "[i] Model type: ${MODEL_TYPE}"
echo "[i] Start: $(date '+%Y-%m-%d %H:%M:%S')"

export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R_llama3x
source ~/.bashrc
# Optional Discord notification (personal; harmless if absent on another machine).
source /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/discord-notif/discord_notif.sh 2>/dev/null || true

# Prompts must already exist under data/prompts/ (run step0 first, CPU & fast):
#   python llm_gen/step0_make_prompts.py --set both
/rds/general/user/yl9422/home/miniforge3/envs/M4R_llama3x/bin/python llm_gen/step1_hidden_states.py \
  --model_type "$MODEL_TYPE" \
  --batch_size 10 \
  --max_new_tokens 256 \
  --hidden_state_dir "data/data-full" \
  --origin_answer_dir "data/data-full"

echo "[i] End: $(date '+%Y-%m-%d %H:%M:%S')"
