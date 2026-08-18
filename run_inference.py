"""
run_inference.py

Genolator V1 Inference Script.

Runs inference with a trained Genolator model that fuses DNA sequence embeddings
(from Evo2) with either protein structure embeddings (PST) or amino acid embeddings
(ESM-2) for multimodal gene function prediction using a fine-tuned LLaMA model.
Select the modality with --embedding_type {pst,esm}.

The script:
1. Loads a test dataset with pre-computed DNA and protein embeddings
2. Loads trained model weights (LLaMA + LoRA, projectors)
3. Runs batched inference to generate functional annotations
4. Saves results with generated text, hidden states, and attention analysis

Usage:
    python run_inference.py \\
        --embedding_type pst \\
        --dataset_path /path/to/test_data.pkl \\
        --output_dir /path/to/outputs \\
        --trained_model_path /path/to/model_weights.pt \\
        --trained_dna_projector_path /path/to/dna_projector.pt \\
        --trained_protein_projector_path /path/to/pst_projector.pt \\
        --hf_token YOUR_HF_TOKEN
"""

import os
import argparse
import logging

import pandas as pd
import torch
import mlflow

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from huggingface_hub import login
from tqdm import tqdm

from utils.data import DEFAULT_DATASET, SPLITS, inference_columns, load_split
from utils.token_projector import GenomicVirtualTokenProjector
from utils.utils import run_and_save_inference
from utils.columns import (
    DNA_EMBEDDING_COLUMN,
    ESM_EMBEDDING_COLUMN,
    PST_EMBEDDING_COLUMN,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------ MODALITY CONFIGURATION -----------------------------------
# The DNA + PST and DNA + ESM models share one architecture and one inference procedure.
# Everything that differs between them lives in this table.

MODALITIES = {
    "pst": {
        "label": "PST",
        "embedding_column": PST_EMBEDDING_COLUMN,  # see utils/columns.py
        "attn_column": "pst_attn",             # results column for the protein attention trace
    },
    "esm": {
        "label": "ESM",
        "embedding_column": ESM_EMBEDDING_COLUMN,
        "attn_column": "esm_attn",
    },
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Genolator inference with Evo2 DNA and protein structure or amino-acid Embeddings'
    )


    # Modality selection
    parser.add_argument('--embedding_type', type=str, required=True,
                        choices=sorted(MODALITIES),
                        help='Protein modality fused with DNA: pst or esm')

    # Evaluation subset
    parser.add_argument('--kinds', type=str, nargs='*', default=None,
                        help='Restrict evaluation to these question kinds '
                             '(e.g. confirmation denial generic). '
                             'Default: evaluate the full dataset.')

    # Dataset
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help='HuggingFace dataset repo id, or a local directory '
                             'holding the splits. Default: the published dataset '
                             f'{DEFAULT_DATASET}')
    parser.add_argument('--split', type=str, default='test', choices=list(SPLITS),
                        help='Split to evaluate. Default: test')
    parser.add_argument('--dataset_revision', type=str, default=None,
                        help='Dataset repo revision (branch, tag or commit sha). '
                             'Pin a commit for an exactly reproducible run.')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Override the split with a local file '
                             '(.parquet or a pickled DataFrame)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for datasets')
    parser.add_argument('--output_suffix', type=str, required=True,
                        help='Suffix for the file naming')
    
    parser.add_argument('--trained_model_path', type=str, required=False,
                        help='Path to the trained genolator model weights')
    parser.add_argument('--trained_dna_projector_path', type=str, required=True,
                        help='Path to the pretrained DNA Projector weights')
    parser.add_argument('--trained_protein_projector_path', type=str, required=True,
                        help='Path to the pretrained protein-side (structure or amino-acid) Projector weights')

    # HuggingFace
    parser.add_argument('--hf_token', type=str, required=True,
                        help='HuggingFace Token to access Llama model')

    # Lora Config
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank (r)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout (0 for Unsloth fast patching)')
    parser.add_argument('--lora_all_target_modules', type=str, default='false',
                        help='LoRA target modules if false only adapters for q_proj and v_proj will be used)')


    parser.add_argument('--subset_size', type=int, default=5000,
                        help='Number of samples to process per batch/iteration')
    parser.add_argument('--num_virtual_tokens', type=int, default=8,
                        help='Number of virtual tokens per projector (must match training)')

    # Embedding scaling
    parser.add_argument('--scale_virtual_embeddings', type=str, default='false',
                    help='Scale virtual token embeddings to match text embedding norms (must match training)')

    return parser.parse_args()

# -------------------------------------------------------------------------------

if __name__ == "__main__":
    # ======================== PARSE ARGUMENTS ========================
    args = parse_args()

    # ======================== CONFIGURATION ========================
    # ======================== MODALITY RESOLUTION ========================
    embedding_type = args.embedding_type
    modality = MODALITIES[embedding_type]
    label = modality['label']
    embedding_column = modality['embedding_column']
    attn_column = modality['attn_column']
    kinds = args.kinds

    # A --dataset_path override wins over --dataset.
    split = args.split
    dataset_revision = args.dataset_revision
    dataset_source = args.dataset_path or args.dataset
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_suffix = args.output_suffix
    
    trained_model_path = args.trained_model_path
    if not trained_model_path:
        logging.info("No model path given. Inference based on trained projectors only.")
    trained_dna_projector_path = args.trained_dna_projector_path
    trained_protein_projector_path = args.trained_protein_projector_path
    
    hf_token = args.hf_token
    
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_all_target_modules = args.lora_all_target_modules.lower() == 'true'

        
    num_virtual_tokens = args.num_virtual_tokens
    subset_size = args.subset_size
    scale_virtual_embeddings = args.scale_virtual_embeddings.lower() == 'true'


    
    # ---------------------------------- DATA LOADING AND SPLITTING ------------------------------------

    # Authenticate to Hugging Face Hub (for model/tokenizer loading)
    login(hf_token)

    # By default the split comes from the published HuggingFace dataset, downloaded
    # and cached by huggingface_hub; --dataset_path reads a local file instead. Only
    # the columns used here are read: both embeddings for the selected modality and
    # the metadata carried into the results table.
    df = load_split(
        dataset_source,
        split,
        columns=inference_columns(embedding_column),
        token=hf_token,
        revision=dataset_revision,
    )

    # Restrict the evaluation set to the requested question kinds. The published {label}
    # run evaluated the full test set (no --kinds), the published ESM run used
    # `--kinds confirmation denial`; passing it explicitly keeps the evaluated subset
    # visible on the command line and recorded in MLflow instead of hidden in a diff.
    if kinds:
        df = df[df.kind.isin(kinds)].copy()

    logging.info(f"Evaluating {len(df)} samples (kinds: {', '.join(kinds) if kinds else 'all'})")
    mlflow.log_params({
        "eval_kinds": ",".join(kinds) if kinds else "all",
        "eval_num_samples": len(df),
        "eval_dataset": dataset_source,
        "eval_split": split,
        "eval_dataset_revision": dataset_revision or "default",
    })

    mlflow.log_metrics({
        "samples": len(df),
    })

    # Load the saved model and projector state_dicts
    if trained_model_path:
        with open(trained_model_path, "rb") as f:
            state_dict_model = torch.load(f, map_location="cuda")

    with open(trained_dna_projector_path, "rb") as f:
        state_dict_dna_projector = torch.load(f, map_location="cuda")

    with open(trained_protein_projector_path, "rb") as f:
        state_dict_protein_projector = torch.load(f, map_location="cuda")

    # ----------------- 4. Model & Projector Initialization -------------------------
    llama_model_name = "ContactDoctor/Bio-Medical-Llama-3-8B"

    # Load model and tokenizer with native HuggingFace
    logging.info(f"Loading model: {llama_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        llama_model_name,
        token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        llama_model_name,
        token=hf_token
    )

    if lora_all_target_modules:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    
    else:
        target_modules = ["q_proj", "v_proj"]
    
    mlflow.log_params({"target_modules": (', ').join(target_modules)})
    

    # Apply LoRA with PEFT (must match training configuration)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    logging.info("LoRA applied to model")
    

    # Instantiate projector modules (must match shape as training!)
    # Get embedding dimensions from data
    dna_dim = df.iloc[0][DNA_EMBEDDING_COLUMN].shape[0]
    protein_dim = df.iloc[0][embedding_column].shape[0]

    mlflow.log_params({
        "dna_embedding_dim": dna_dim,
        f"{embedding_type}_embedding_dim": protein_dim,
    })

    # Instantiate projector modules (must match shape as training!)
    dna_projector = GenomicVirtualTokenProjector(
        embedding_dim=dna_dim,
        llama_hidden_size=model.config.hidden_size,
        num_virtual_tokens=num_virtual_tokens,
    )

    protein_projector = GenomicVirtualTokenProjector(
        embedding_dim=protein_dim,
        llama_hidden_size=model.config.hidden_size,
        num_virtual_tokens=num_virtual_tokens,
    )
    
    if trained_model_path:
        print("#"*60)
        print("This is my model state dict: ")
        for key in state_dict_model.keys():
            print(key)
        print("#"*60)
        # Load the best-trained weights for the model and both projectors
        model.load_state_dict(state_dict_model)

    dna_projector.load_state_dict(state_dict_dna_projector)
    protein_projector.load_state_dict(state_dict_protein_projector)

    # Set to evaluation mode and device (CUDA/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dna_projector = dna_projector.to(device).eval()
    protein_projector = protein_projector.to(device).eval()
    
    model.set_attn_implementation('eager')


    # ----------------- 5. Inference/Generation: Chunked Batching Loop ----------------
    output_file_paths = []  # Track all output files for final concatenation

    for _, start in tqdm(enumerate(range(0, df.shape[0], subset_size))):
        print(f"From: {start} to {start+subset_size}")
        iteration = start + subset_size

        # Avoid recomputation: skip output file if it already exists
        output_file_name = f"inference_genolator_v1_unsloth_{embedding_type}_{iteration}.pkl"
        output_file_path = os.path.join(output_dir, output_file_name)
        output_file_paths.append(output_file_path)

        if os.path.isfile(output_file_path):
            print("Iteration does already exist. Continue with next iteration.")
            continue

        subset = df.iloc[
            start : start + subset_size
        ].copy()  # Copy avoids chained assignment issues
        run_and_save_inference(
            subset,
            tokenizer,
            model,
            dna_projector,
            protein_projector,
            device,
            output_file_path,
            scale_virtual_embeddings=scale_virtual_embeddings,
            protein_embedding_column=embedding_column,
            protein_attn_column=attn_column,
        )
        torch.cuda.empty_cache()  # Explicitly clear CUDA memory after each batch to prevent OOM

    # ----------------- 6. Concatenate All Subsets into Single DataFrame ----------------
    if len(output_file_paths) > 1:
        logging.info(f"Concatenating {len(output_file_paths)} subset files into single DataFrame...")

        dfs_to_concat = []
        for file_path in output_file_paths:
            if os.path.isfile(file_path):
                dfs_to_concat.append(pd.read_pickle(file_path))
            else:
                logging.warning(f"Expected file not found: {file_path}")

        if dfs_to_concat:
            df_combined = pd.concat(dfs_to_concat, ignore_index=True)
            combined_output_path = os.path.join(output_dir, f"inference_genolator_v1_unsloth_{embedding_type}_{output_suffix}.pkl")
            df_combined.to_pickle(combined_output_path)
            logging.info(f"Combined DataFrame saved to: {combined_output_path}")
            logging.info(f"Total samples: {len(df_combined)}")

            mlflow.log_params({
                "total_inference_samples": len(df_combined),
                "num_subset_files": len(dfs_to_concat),
            })
    else:
        logging.info("Only one subset file created, no concatenation needed.")
