#!/bin/bash
set -euo pipefail

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R_llama3x_vllm

cd /rds/general/user/yl9422/home/files/M4R-Elly

# Reduce import contamination from user site-packages.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# PBS sets CUDA_VISIBLE_DEVICES to a GPU UUID, which new vLLM tries to int() and crashes on.
# Single-GPU interactive session -> pin to local index 0.
export CUDA_VISIBLE_DEVICES=0

# FlashInfer sampling would JIT-compile a CUDA kernel (needs nvcc, absent on compute nodes).
# Use the PyTorch-native sampler instead (no effect on temperature=0.0 greedy decoding).
export VLLM_USE_FLASHINFER_SAMPLER=0

PROMPT_FILE="/rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/chat/prompts/linear_probe.txt"
OUTPUT_DIR="/rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/chat/outputs"
MODELS="3_1_8b,3_2_3b,3_2_1b"    # all loaded once; pick per round via '#model: <name>' on line 1 of the prompt file

python \
  /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/chat/interactive/run_prompt_loop_vllm.py \
  --models "${MODELS}" \
  --gpu-frac "3_1_8b=0.45,3_2_3b=0.25,3_2_1b=0.15" \
  --prompt-file "${PROMPT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-tokens 512 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-model-len 4096 \
  --tensor-parallel-size 1
