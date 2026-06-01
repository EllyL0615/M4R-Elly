#!/bin/bash

BASE_DIR="/rds/general/user/yl9422/home/files/models"

MODELS=(
    # "meta-llama/Llama-2-7b-chat-hf"    # in the Paper
    # "meta-llama/Llama-2-13b-chat-hf"    # in the Paper
    # "meta-llama/Meta-Llama-3-8B"    # in the Paper
    # "meta-llama/Meta-Llama-3-8B-Instruct"    # in the Paper
    # "DeepSeek-R1-Distill-Qwen-7B"
    # "DeepSeek-Prover-V1.5-RL"
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
)

for MODEL in "${MODELS[@]}"; do
    LOCAL_NAME=$(basename "$MODEL")
    LOCAL_DIR="$BASE_DIR/$LOCAL_NAME"

    echo "======================================="
    echo "Downloading $MODEL  ->  $LOCAL_DIR"
    echo "======================================="

    huggingface-cli download "$MODEL" \
        --local-dir "$LOCAL_DIR"
done
