#!/bin/bash
set -euo pipefail

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R

cd /rds/general/user/yl9422/home/files/M4R-Elly

# Reduce import contamination from user site-packages.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# vLLM 0.4.2 workaround for some xformers/triton mismatches.
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

PROMPT_FILE="/rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/chat/prompts/example_prompt.txt"
OUTPUT_DIR="/rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/chat/outputs"
MODEL_TYPE="3_8b"    # choose from "2_7b", "2_13b", "3_8b_instruct", "3_8b", "deepseek"

/rds/general/user/yl9422/home/miniforge3/envs/M4R/bin/python \
  /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/chat/interactive/run_prompt_loop_vllm.py \
  --model-type "${MODEL_TYPE}" \
  --prompt-file "${PROMPT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-tokens 512 \
  --temperature 0.0 \
  --top-p 1.0 \
  --tensor-parallel-size 1
