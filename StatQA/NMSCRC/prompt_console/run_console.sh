#!/bin/bash
set -euo pipefail

# Interactive multi-model prompt console. Run inside an interactive GPU session:
#   qsub -I -l select=1:ncpus=8:mem=64gb:ngpus=1 -l walltime=02:00:00
#   bash NMSCRC/prompt_console/run_console.sh
# Loading the three resident models takes ~5 min (incl. torch.compile).

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate M4R_llama3x_vllm

CONSOLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CONSOLE_DIR"

# Reduce import contamination from user site-packages.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# PBS sets CUDA_VISIBLE_DEVICES to a GPU UUID, which new vLLM tries to int() and crashes on.
# Single-GPU interactive session -> pin to local index 0.
export CUDA_VISIBLE_DEVICES=0

# FlashInfer sampling would JIT-compile a CUDA kernel (needs nvcc, absent on compute nodes).
# Use the PyTorch-native sampler instead (no effect on temperature=0.0 greedy decoding).
export VLLM_USE_FLASHINFER_SAMPLER=0

MODELS="3_1_8b,3_2_3b,3_2_1b"    # all loaded once; pick per round via '#model: <name>' on line 1 of the prompt file

python run_prompt_loop_vllm.py \
  --models "${MODELS}" \
  --gpu-frac "3_1_8b=0.45,3_2_3b=0.25,3_2_1b=0.15" \
  --prompt-file "${CONSOLE_DIR}/prompts/linear_probe.txt" \
  --output-dir "${CONSOLE_DIR}/outputs" \
  --max-tokens 512 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-model-len 4096 \
  --tensor-parallel-size 1
