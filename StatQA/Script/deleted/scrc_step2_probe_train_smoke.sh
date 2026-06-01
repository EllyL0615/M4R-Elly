#!/bin/bash
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:15:00
#PBS -N probe_train_smoke
#PBS -o /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/SCRC/outputs/step2_smoke
#PBS -e /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/SCRC/outputs/step2_smoke

set -e

cd /rds/general/user/yl9422/home/files/M4R-Elly/StatQA
mkdir -p SCRC/outputs/step2_smoke

echo "[i] Host: $(hostname)"
echo "[i] Start: $(date '+%Y-%m-%d %H:%M:%S')"

export PYTHONHASHSEED=42

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R
source ~/.bashrc
source /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/discord-notif/discord_notif.sh

/rds/general/user/yl9422/home/miniforge3/envs/M4R/bin/python SCRC/2_probe_train.py \
  --model_name "llama3_8b_smoke" \
  --outputs_dir "SCRC/outputs/step2_smoke" \
  --saved_probe_dir "SCRC/outputs/step2_smoke" \
  --epochs 1 \
  --batch_size 1024 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --patience 1 \
  --val_ratio 0.1 \
  --device cpu \
  --use_pos_weight \
  --save_labels

echo "[i] End: $(date '+%Y-%m-%d %H:%M:%S')"
