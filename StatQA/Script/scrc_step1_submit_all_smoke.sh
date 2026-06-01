#!/bin/bash
# Submit Step-1 SMOKE validation jobs (3 rows/split) for the LLaMA models.
#
# Usage:
#   bash Script/scrc_step1_submit_all_smoke.sh                 # all 3 models
#   bash Script/scrc_step1_submit_all_smoke.sh 3_1_8b 3_2_3b   # a subset
#
# One model per job. Smoke outputs are auto-named per model and land in
# SCRC/data/data-smoke/ (isolated from the full-run data in SCRC/data/data-full/).
set -e

JOB="/rds/general/user/yl9422/home/files/M4R-Elly/StatQA/Script/scrc_step1_hs_gen_smoke.sh"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(3_1_8b 3_2_1b 3_2_3b)
fi

for MT in "${MODELS[@]}"; do
  jid=$(qsub -N "sm_${MT}" -v MODEL_TYPE="${MT}" "${JOB}")
  echo "[+] submitted smoke model_type=${MT}  ->  ${jid}"
done

echo "[i] Monitor with: qstat -u yl9422"
