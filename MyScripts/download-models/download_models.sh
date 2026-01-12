#!/bin/bash

BASE_DIR="/rds/general/user/yl9422/home/files/models"

MODELS=(
    # "meta-llama/Llama-2-7b-chat-hf"    # in the Paper
    # "meta-llama/Llama-2-13b-chat-hf"    # in the Paper
    # "meta-llama/Meta-Llama-3-8B"    # in the Paper
    # "meta-llama/Meta-Llama-3-8B-Instruct"    # in the Paper
)

for MODEL in "${MODELS[@]}"; do
    LOCAL_NAME=$(basename "$MODEL")
    LOCAL_DIR="$BASE_DIR/$LOCAL_NAME"

    echo "======================================="
    echo "Downloading $MODEL  ->  $LOCAL_DIR"
    echo "======================================="

    hf download "$MODEL" \
        --local-dir "$LOCAL_DIR"
done
