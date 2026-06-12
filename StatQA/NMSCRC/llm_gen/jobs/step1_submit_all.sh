#!/bin/bash
# Submit Step-1 hidden-state generation jobs for the LLaMA models (one job per model).
#
# Usage (from the NMSCRC root):
#   bash llm_gen/jobs/step1_submit_all.sh                 # all 3 models
#   bash llm_gen/jobs/step1_submit_all.sh 3_1_8b 3_2_3b   # a subset
#
# NOTE: run with `bash` (this is a submitter that calls qsub), NOT `qsub`.
# Outputs are auto-named per model: llama3_1_8b_* / llama3_2_1b_* / llama3_2_3b_*
set -e

JOB="/rds/general/user/yl9422/home/files/M4R-Elly/StatQA/NMSCRC/llm_gen/jobs/step1_hs_gen.sh"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(3_1_8b 3_2_1b 3_2_3b)
fi

for MT in "${MODELS[@]}"; do
  jid=$(qsub -N "nmscrc_hs_${MT}" -v MODEL_TYPE="${MT}" "${JOB}")
  echo "[+] submitted model_type=${MT}  ->  ${jid}"
done

echo "[i] Monitor with: qstat -u yl9422"
