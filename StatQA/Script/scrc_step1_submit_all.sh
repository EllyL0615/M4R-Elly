#!/bin/bash
# Submit Step-1 hidden-state generation jobs for the LLaMA models.
#
# Usage:
#   bash Script/scrc_step1_submit_all.sh                 # all 3 models
#   bash Script/scrc_step1_submit_all.sh 3_1_8b 3_2_3b   # a subset
#
# One model per job (a failure in one does not affect the others). Outputs are
# auto-named per model: llama3_1_8b_* / llama3_2_1b_* / llama3_2_3b_*
set -e

JOB="/rds/general/user/yl9422/home/files/M4R-Elly/StatQA/Script/scrc_step1_hs_gen.sh"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(3_1_8b 3_2_1b 3_2_3b)
fi

for MT in "${MODELS[@]}"; do
  jid=$(qsub -N "hs_${MT}" -v MODEL_TYPE="${MT}" "${JOB}")
  echo "[+] submitted model_type=${MT}  ->  ${jid}"
done

echo "[i] Monitor with: qstat -u yl9422"
