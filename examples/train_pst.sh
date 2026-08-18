#!/usr/bin/env bash
# Reproduces the published DNA + PST model:
#   genolator_v1_dna_and_pst/genolator_dna_and_pst.pt
#
# Requires HF_TOKEN in the environment (gated Llama-3 weights).
set -euo pipefail

# The splits are pulled from the published HuggingFace dataset and cached by
# huggingface_hub. Set DATA_DIR to train on local files instead.
DATASET="${DATASET:-CHGGM-Aachen/genolator-v1-qa}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/genolator_pst_v1}"

mkdir -p "$OUTPUT_DIR"

python3 ../train_genolator.py \
    --embedding_type pst \
    --dataset "${DATA_DIR:-$DATASET}" \
    --output_dir "$OUTPUT_DIR" \
    --hf_token "$HF_TOKEN" \
    --batch_size 8 \
    --num_epochs 10 \
    --patience 3 \
    --num_virtual_tokens 8 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_all_target_modules true \
    --scale_virtual_embeddings true
