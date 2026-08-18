#!/usr/bin/env bash
# Generation against the published DNA + ESM checkpoint.
#
# CHECKPOINT_DIR must contain the three artefacts written by training:
#   genolator_dna_and_esm.pt, dna_projector.pt, esm_projector.pt
#
# The published ESM run evaluated only confirmation and denial questions
# (generic was excluded), reproduced here by --kinds.
#
# The LoRA flags MUST match training: the published model used all seven
# target modules and scaled virtual embeddings.
set -euo pipefail

# The test split is pulled from the published HuggingFace dataset and cached by
# huggingface_hub. Set DATA_DIR to evaluate a local file instead.
DATASET="${DATASET:-CHGGM-Aachen/genolator-v1-qa}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the genolator_v1_dna_and_esm model folder}"
OUTPUT_DIR="${OUTPUT_DIR:-./inference/genolator_esm_v1}"

mkdir -p "$OUTPUT_DIR"

python3 ../run_inference.py \
    --embedding_type esm \
    --dataset "${DATA_DIR:-$DATASET}" \
    --split test \
    --output_dir "$OUTPUT_DIR" \
    --output_suffix "test" \
    --trained_model_path "$CHECKPOINT_DIR/genolator_dna_and_esm.pt" \
    --trained_dna_projector_path "$CHECKPOINT_DIR/dna_projector.pt" \
    --trained_protein_projector_path "$CHECKPOINT_DIR/esm_projector.pt" \
    --hf_token "$HF_TOKEN" \
    --num_virtual_tokens 8 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_all_target_modules true \
    --scale_virtual_embeddings true \
    --subset_size 200 \
    --kinds confirmation denial
