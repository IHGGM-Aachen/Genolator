import torch
from torch.utils.data import DataLoader
import argparse

from unsloth import FastLanguageModel
from transformers import AutoTokenizer

import os

from huggingface_hub import login
from tqdm import tqdm

import mlflow

import uuid

import logging

from utils.data import DEFAULT_DATASET, load_split, subset_splits, training_columns
from utils.token_projector import GenomicVirtualTokenProjector
from utils.qa_dataset import GenomicQADatasetESM, GenomicQADatasetPST
from utils.utils import custom_collate, evaluate_model
from utils.columns import (
    DNA_EMBEDDING_COLUMN,
    ESM_BATCH_KEY,
    ESM_EMBEDDING_COLUMN,
    PST_BATCH_KEY,
    PST_EMBEDDING_COLUMN,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


"""
train_genolator.py

Genolator V1 Training Script - unified across protein modalities.

Fine-tunes a Llama-3 model for genomics Q&A on multimodal input, fusing DNA (Evo2)
embeddings with either protein structure (PST) or amino acid (ESM-2) embeddings via
virtual token projection. Select the modality with --embedding_type {pst,esm}.

Training Strategy:
    Train LLaMA LoRA adapters + both token projectors against the task loss only.

Architecture Flow:
    Token Projectors (8 tokens/modality) -> Concatenate with Text -> LLaMA ->
    Task Loss (next-token prediction)

Requirements:
    HuggingFace Transformers, PEFT, unsloth, mlflow
    PyTorch with GPU support recommended
"""


# ------------------------------ MODALITY CONFIGURATION -----------------------------------
# The DNA + PST and DNA + ESM models share one architecture and one training procedure.
# Everything that differs between them lives in this table.

MODALITIES = {
    "pst": {
        "label": "PST",
        "embedding_column": PST_EMBEDDING_COLUMN,  # see utils/columns.py
        "batch_key": PST_BATCH_KEY,                # key custom_collate emits for it
        "dataset_cls": GenomicQADatasetPST,
        "model_filename": "genolator_dna_and_pst.pt",
        "projector_filename": "pst_projector.pt",
    },
    "esm": {
        "label": "ESM",
        "embedding_column": ESM_EMBEDDING_COLUMN,
        "batch_key": ESM_BATCH_KEY,
        "dataset_cls": GenomicQADatasetESM,
        "model_filename": "genolator_dna_and_esm.pt",
        "projector_filename": "esm_projector.pt",
    },
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune multimodal Llama Model with Evo2 and protein structure or amino-acid Embeddings'
    )

    # Modality selection
    parser.add_argument('--embedding_type', type=str, required=True,
                        choices=sorted(MODALITIES),
                        help='Protein modality fused with DNA: pst or esm')

    # Dataset
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help='HuggingFace dataset repo id, or a local directory '
                             'holding the splits. Default: the published dataset '
                             f'{DEFAULT_DATASET}')
    parser.add_argument('--dataset_revision', type=str, default=None,
                        help='Dataset repo revision (branch, tag or commit sha). '
                             'Pin a commit for an exactly reproducible run.')

    # Per-split overrides: point these at your own files to train on other data.
    # Each accepts a .parquet file or a pickled DataFrame.
    parser.add_argument('--train_path', type=str, default=None,
                        help='Override the training split with a local file')
    parser.add_argument('--val_path', type=str, default=None,
                        help='Override the validation split with a local file')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Override the test split with a local file')

    # Smoke tests: cap the splits instead of pulling the full dataset through.
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap each split to this many rows, applied after the kind '
                             'filter. The training split is capped per group so every '
                             'epoch of the group rotation still has data. Intended for '
                             'smoke tests; leave unset to use the full splits.')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for token projectors and Llama model')

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

    # Hyperparameters
    parser.add_argument('--num_virtual_tokens', type=int, default=8,
                        help='Number of virtual tokens per projector')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for the LLaMA LoRA adapters, and the default '
                             'for the token projectors')
    parser.add_argument('--projector_learning_rate', type=float, default=None,
                        help='Separate learning rate for the DNA/protein token projectors. '
                             'Default: same as --learning_rate. Useful with '
                             '--train_token_projectors_only, where a higher rate helps.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience (epochs without improvement)')
    
    # Embedding scaling
    parser.add_argument('--scale_virtual_embeddings', type=str, default='false',
                    help='Scale virtual token embeddings to match text embedding norms')

    # Training mode
    parser.add_argument('--train_token_projectors_only', type=str, default='false',
                    help='If true, freeze LLaMA (incl. LoRA) and only train token projectors')

    return parser.parse_args()


if __name__ == "__main__":
    # ======================== PARSE ARGUMENTS ========================
    args = parse_args()

    # ======================== MODALITY RESOLUTION ========================
    embedding_type = args.embedding_type
    modality = MODALITIES[embedding_type]
    label = modality['label']
    embedding_column = modality['embedding_column']
    batch_key = modality['batch_key']
    dataset_cls = modality['dataset_cls']
    model_filename = modality['model_filename']
    projector_filename = modality['projector_filename']

    # ======================== CONFIGURATION ========================
    dataset = args.dataset
    dataset_revision = args.dataset_revision
    # A --*_path override wins over --dataset for that split.
    train_source = args.train_path or dataset
    val_source = args.val_path or dataset
    test_source = args.test_path or dataset
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    hf_token = args.hf_token
    
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_all_target_modules = args.lora_all_target_modules.lower() == 'true'
    
    num_virtual_tokens = args.num_virtual_tokens
    learning_rate = args.learning_rate
    projector_learning_rate = (args.projector_learning_rate
                               if args.projector_learning_rate is not None
                               else learning_rate)
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    patience = args.patience
    max_samples = args.max_samples
    scale_virtual_embeddings = args.scale_virtual_embeddings.lower() == 'true'
    train_token_projectors_only = args.train_token_projectors_only.lower() == 'true'

    # Start MLflow run (local store in the working directory by default;
    # no tracking server required -- see README)
    mlflow.start_run()
    
    

    # Log hyperparameters
    mlflow.log_params({
        "num_virtual_tokens": num_virtual_tokens,
        "learning_rate_llama": learning_rate,
        "learning_rate_token_projectors": projector_learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "patience": patience,
        "lora_dropout": lora_dropout,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "unsloth": "True",
        "max_sequence_length": "8192",
        "scale_virtual_embeddings": str(scale_virtual_embeddings),
        "train_token_projectors_only": str(train_token_projectors_only),
    })

    logging.info("=" * 60)
    logging.info("Loading Datasets")
    logging.info("=" * 60)
    # --------------------------- LOGIN/LOAD DATASET --------------------------------------------

    # The released splits are already gene-disjoint and the training split is already
    # class-balanced per `group`, so there is no shuffling, filtering or splitting step
    # here -- the splits are read as published.

    # ---------------------------------- DATA LOADING AND SPLITTING ------------------------------------

    # Authenticate to Hugging Face Hub (for model/tokenizer loading)
    login(hf_token)

    # Load the splits (they carry the precomputed embeddings). By default they come
    # from the published HuggingFace dataset, downloaded and cached by huggingface_hub;
    # --train_path/--val_path/--test_path override an individual split with a local file.
    #
    # Only the columns this run touches are read: the two embeddings for the selected
    # modality, the prompt/response text, `kind`, and `group` on train. The raw sequence
    # columns are the bulk of the dataset on disk and are never used for training.
    df_train = load_split(
        train_source,
        "train",
        columns=training_columns(embedding_column, with_group=True),
        token=hf_token,
        revision=dataset_revision,
    )

    df_val = load_split(
        val_source,
        "validation",
        columns=training_columns(embedding_column, with_group=False),
        token=hf_token,
        revision=dataset_revision,
    )

    df_test = load_split(
        test_source,
        "test",
        columns=training_columns(embedding_column, with_group=False),
        token=hf_token,
        revision=dataset_revision,
    )
    
    kinds = ["confirmation", "denial", "generic"]
    df_train = df_train[df_train.kind.isin(kinds)]
    df_val = df_val[df_val.kind.isin(kinds)]
    df_test = df_test[df_test.kind.isin(kinds)]

    # Optional cap for smoke tests; a no-op unless --max_samples is given.
    df_train, df_val, df_test = subset_splits(df_train, df_val, df_test, max_samples)
    if max_samples is not None:
        mlflow.log_params({"max_samples": max_samples})
    
    
    logging.info(f"Training samples: {len(df_train)}, Validation samples: {len(df_val)}, Test samples: {len(df_test)}")
    mlflow.log_metrics({
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
    })
    # Record where the data came from, so a run can be traced back to its splits.
    mlflow.log_params({
        "dataset_train": train_source,
        "dataset_validation": val_source,
        "dataset_test": test_source,
        "dataset_revision": dataset_revision or "default",
    })

    
    logging.info("=" * 60)
    logging.info("Loading and setting up Models")
    logging.info("=" * 60)

    logging.info("Initializing Llama Model using unsloth...")
    # -------------------------- MODEL LOADING/CONSTRUCTION -------------------------------------------------------
    # Load LLAMA model and tokenizer
    # Initialize PEFT/LoRA wrapper for parameter-efficient fine-tuning.

    # Instantiate projector networks corresponding to each modality.
    # All projectors and model params are included in optimizer.
    llama_model_name = "ContactDoctor/Bio-Medical-Llama-3-8B"
    hf_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

    # The following is only used if original LLAMA is used for training
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=llama_model_name,
        max_seq_length=8192,  # Adjust based on your sequence lengths
        dtype=None,  # Auto-detect (BF16 for H100)
        load_in_4bit=False,  # Don't use 4-bit with H100s (you have enough memory)
        token=hf_token,
    )
    
    # Force padding token to match standard HuggingFace behavior (consistent with V1)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    logging.info(f"Tokenizer configured: pad_token={tokenizer.pad_token}, pad_token_id={tokenizer.pad_token_id}")

    mlflow.log_params({
    "hf_pad_token": hf_tokenizer.pad_token,
    "hf_pad_token_id": hf_tokenizer.pad_token_id,
    "hf_padding_side": hf_tokenizer.padding_side,
    "hf_eos_token": hf_tokenizer.eos_token,
    "hf_eos_token_id": hf_tokenizer.eos_token_id,
    "hf_bos_token": hf_tokenizer.bos_token,
    "hf_bos_token_id": hf_tokenizer.bos_token_id,
    "hf_model_max_length": hf_tokenizer.model_max_length,

    "unsloth_pad_token": tokenizer.pad_token,
    "unsloth_pad_token_id": tokenizer.pad_token_id,
    "unsloth_padding_side": tokenizer.padding_side,
    "unsloth_eos_token": tokenizer.eos_token,
    "unsloth_eos_token_id": tokenizer.eos_token_id,
    "unsloth_bos_token": tokenizer.bos_token,
    "unsloth_bos_token_id": tokenizer.bos_token_id,
    "unsloth_model_max_length": tokenizer.model_max_length,
})
    
    if lora_all_target_modules:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    
    else:
        target_modules = ["q_proj", "v_proj"]
    
    mlflow.log_params({"target_modules": (', ').join(target_modules)})
    
    # Apply Unsloth's optimized LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )
    
    model_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model params: {model_params:,}")

    logging.info("Initializing token projectors...")

    # ---- DATASET GROUPS (for balanced training/oversampling, see earlier prep) ----
    groups = [1, 2, 3]

    # --------------------- VIRTUAL TOKEN PROJECTORS (MULTIMODAL FUSION LAYERS) --------------------
    # Each projector takes a modality-specific embedding and produces a matrix
    # of virtual token embeddings with the right shape for Llama's transformer.

    # Get embedding dimensions from data
    dna_dim = df_train.iloc[0][DNA_EMBEDDING_COLUMN].shape[0]
    protein_dim = df_train.iloc[0][embedding_column].shape[0]

    mlflow.log_params({
        "dna_embedding_dim": dna_dim,
        f"{embedding_type}_embedding_dim": protein_dim,
    })

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
    
    logging.info(f"DNA projector: {dna_dim} -> {num_virtual_tokens} x {model.config.hidden_size}")
    logging.info(f"{label} projector: {protein_dim} -> {num_virtual_tokens} x {model.config.hidden_size}")

    # Log model parameter counts
    dna_params = sum(p.numel() for p in dna_projector.parameters())
    protein_params = sum(p.numel() for p in protein_projector.parameters())
    logging.info(f"DNA projector params: {dna_params:,}")
    logging.info(f"{label} projector params: {protein_params:,}")

    # -------------- PARAMETER COUNTS ------------------------------------
    mlflow.log_params({
        "model_params": model_params,
        "dna_projector_params": dna_params,
        f"{embedding_type}_projector_params": protein_params,
        "total_params": model_params + dna_params + protein_params,
    })

    # -------------- OPTIMIZER SETUP ------------------------------------
    if train_token_projectors_only:
        # Freeze all LLaMA parameters (including LoRA adapters)
        for param in model.parameters():
            param.requires_grad = False
        trainable_params = [{'params': dna_projector.parameters(), 'lr': projector_learning_rate},
                            {'params': protein_projector.parameters(), 'lr': projector_learning_rate}]
        logging.info("Optimizer training: Token Projectors ONLY (LLaMA frozen)")
        mlflow.log_params({"Trainable Params": f"DNA Projector, {label} Projector"})
    else:
        trainable_params = [{'params': model.parameters(), 'lr': learning_rate},
                            {'params': dna_projector.parameters(), 'lr': projector_learning_rate},
                            {'params': protein_projector.parameters(), 'lr': projector_learning_rate}]
        logging.info("Optimizer training: LLaMA LoRA + Token Projectors")
        mlflow.log_params({"Trainable Params": f"Llama, DNA Projector, {label} Projector"})

    optimizer = torch.optim.AdamW(trainable_params)

    # -------------- DEVICE (CPU/GPU) CONFIGURATION ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        
    logging.info("=" * 60)
    logging.info("Starting training...")
    logging.info("=" * 60)

    # Move everything to the correct device
    model = model.to(device)
    dna_projector = dna_projector.to(device)
    protein_projector = protein_projector.to(device)

    # Set modules to training/eval mode
    model.train()

    # When only training projectors, disable dropout in LLaMA to get
    # deterministic outputs (equivalent to model.eval() for dropout,
    # but avoids Unsloth issues with loss computation in eval mode)
    if train_token_projectors_only:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    # Trainable projectors → use train mode
    dna_projector.train()
    protein_projector.train()

    # -------------- EARLY STOPPING LOGIC -------------------------------
    best_val_loss = float("inf")  # Best model performance so far
    patience_counter = 0  # Current 'wait' without improvement

    # -------------- VALIDATION SETUP -----------------------------------
    # Prepare GenomicQADataset for validation (uses tokenization + embedding references)
    #df_val = df_val.iloc[0:1280]
    
    val_dataset = dataset_cls(
        df_val, tokenizer, num_virtual_tokens=num_virtual_tokens
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Use same batch size as training for fair comparison
        shuffle=False,
        collate_fn=custom_collate,  # Custom collate bundles modality features and input tokens
    )

    # -------------------------- TRAINING LOOP --------------------------------------------------

    # Use MLFlow for experiment tracking
    # Epoch loop: Alternate which training "group" to use (to balance clusters)
    #   For each batch:
    #      - Project embeddings into virtual tokens
    #      - Concat them with text token embeddings
    #      - Adjust label/attn mask for new tokens
    #      - Forward, loss, backward, optimizer step

    # Log metrics, implement patience-based early stopping
    # On improvement: save best model/projectors to path

    # ------------------- EPOCH LOOP ---------------------------------
    for epoch in range(num_epochs):
        # Use group-based training for balancing: rotate which set of examples are used each epoch
        group_idx = epoch % 3  # Alternate between group 1, 2, 3

        logging.info(
            "Training on subset based on group assignments: {}".format(
                groups[group_idx]
            )
        )

        # Select corresponding training data for this epoch
        df_train_subset = df_train[df_train["group"] == groups[group_idx]]
        
        #df_train_subset = df_train_subset.iloc[0:1280]

        # Build dataset and loader for this specific group
        train_dataset = dataset_cls(
            df_train_subset, tokenizer, num_virtual_tokens=num_virtual_tokens
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate,
        )

        total_train_loss = 0
        epoch_dna_scale_factors = []
        epoch_protein_scale_factors = []
        batch_idx = 0
        # ------------------ TRAINING LOOP (BATCHES) --------------------------------
        for batch in tqdm(train_loader):
            # Move text tensors to device
            input_ids = batch["input_ids"].to(device)  # [batch, seq_len]
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Each embedding list: one per batch element, align with virtual token projectors
            dna_emb_list = batch[
                "nucleotide_emb"
            ]  # List of [emb_dim] or None
            protein_emb_list = batch[batch_key]

            # ---- Modality Projection: Virtual Token Construction ----
            # (1) dna
            vtoken_dna_embeds = []
            for emb in dna_emb_list:
                if emb is None:
                    # Use learnable null embedding if no data
                    vtoken_dna_embeds.append(
                        dna_projector().to(device)
                    )
                else:
                    # Project real embedding to virtual tokens
                    vtoken_dna_embeds.append(
                        dna_projector(emb.unsqueeze(0).to(device))
                    )  # shape [1, n_tokens, hidden]
            vtoken_dna_embeds = torch.cat(
                vtoken_dna_embeds, dim=0
            )  # [batch, n_tokens, hidden]

            # (2) Amino acid
            vtoken_protein_embeds = []
            for emb in protein_emb_list:
                if emb is None:
                    vtoken_protein_embeds.append(protein_projector().to(device))
                else:
                    vtoken_protein_embeds.append(
                        protein_projector(emb.unsqueeze(0).to(device))
                    )
            vtoken_protein_embeds = torch.cat(vtoken_protein_embeds, dim=0)


            # ---- Standard Language Model Embedding look-up for natural language tokens ---
            text_embeds = model.get_input_embeddings()(
                input_ids
            )  # [batch, seq_len, hidden]

            # ---- Scale virtual token embeddings to match text embedding norms ----
            if scale_virtual_embeddings:
                text_norm = text_embeds.norm(dim=-1, keepdim=True).mean()
                dna_norm = vtoken_dna_embeds.norm(dim=-1, keepdim=True).mean()
                protein_norm = vtoken_protein_embeds.norm(dim=-1, keepdim=True).mean()
                dna_scale_factor = (text_norm / (dna_norm + 1e-8)).item()
                protein_scale_factor = (text_norm / (protein_norm + 1e-8)).item()
                vtoken_dna_embeds = vtoken_dna_embeds * dna_scale_factor
                vtoken_protein_embeds = vtoken_protein_embeds * protein_scale_factor
                epoch_dna_scale_factors.append(dna_scale_factor)
                epoch_protein_scale_factors.append(protein_scale_factor)
                if batch_idx == 0 or (batch_idx + 1) % 200 == 0:
                    global_step = epoch * len(train_loader) + batch_idx
                    mlflow.log_metrics({
                        "batch_dna_scale_factor": dna_scale_factor,
                        f"batch_{embedding_type}_scale_factor": protein_scale_factor,
                    }, step=global_step)

            # ---- Concatenate everything: [virtual dna] + [virtual pst] + [tokens] ----
            inputs_embeds = torch.cat(
                [
                    vtoken_dna_embeds,
                    vtoken_protein_embeds,
                    text_embeds,
                ],
                dim=1,
            )  # [batch, n_tokens*3 + seq_len, hidden]

            # ---- Adjust label and attention-mask tensors to account for new prepend tokens ----
            num_all_virtuals = (
                num_virtual_tokens * 2
            )  # Number of non-language virtual tokens
            virtual_pad = torch.full(
                (labels.size(0), num_all_virtuals),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )  # -100 for ignore in LM loss
            labels = torch.cat([virtual_pad, labels], dim=1)
            virtual_attn = torch.ones(
                (labels.size(0), num_all_virtuals),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([virtual_attn, attention_mask], dim=1)

            # Debug check: Ensure all input shapes now line up
            assert (
                inputs_embeds.shape[1] == labels.shape[1] == attention_mask.shape[1]
            ), f"{inputs_embeds.shape}, {labels.shape}, {attention_mask.shape}"

            # ---- Forward through LLaMA ----
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

            # ---- Compute total loss ----
            # Main training uses ONLY task loss (next-token prediction for QA)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            batch_idx += 1

        # Compute average epoch training loss
        total_train_loss = total_train_loss / len(train_loader)
        logging.info(f"Train loss after epoch {epoch}: {total_train_loss:.4f}")

        # Log scale factors per epoch
        if scale_virtual_embeddings and epoch_dna_scale_factors:
            avg_dna_scale = sum(epoch_dna_scale_factors) / len(epoch_dna_scale_factors)
            avg_protein_scale = sum(epoch_protein_scale_factors) / len(epoch_protein_scale_factors)
            mlflow.log_metrics({
                "dna_scale_factor": avg_dna_scale,
                f"{embedding_type}_scale_factor": avg_protein_scale,
            }, step=epoch)

        # ----------------- CROSS-VALIDATION ---------------------------------
        val_loss = evaluate_model(
            model,
            dna_projector,
            protein_projector,
            val_loader,
            device,
            scale_virtual_embeddings,
            protein_batch_key=batch_key,
        )
        logging.info(f"Validation loss after epoch {epoch}: {val_loss:.4f}")

        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_loss": total_train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss if val_loss >= best_val_loss else val_loss,
            "patience_counter": patience_counter,
        }, step=epoch)

        # --------- EARLY STOPPING: Track best validation loss, save checkpoints if improving ------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logging.info("New best validation loss! Saving models and projectors...")

            if not train_token_projectors_only:
                torch.save(model.state_dict(),
                           os.path.join(output_dir, model_filename))
            torch.save(dna_projector.state_dict(),
                       os.path.join(output_dir, "dna_projector.pt"))
            torch.save(protein_projector.state_dict(),
                       os.path.join(output_dir, projector_filename))
            
             # Save config for later loading
            import json
            config = {
                "dna_embedding_dim": dna_dim,
                f"{embedding_type}_embedding_dim": protein_dim,
                "num_virtual_tokens": num_virtual_tokens,
                "llama_hidden_size": model.config.hidden_size,
                "learning_rate_llama": learning_rate,
                "learning_rate_token_projectors": projector_learning_rate,
                "best_val_loss": best_val_loss,
                "epoch": epoch + 1,
                "train_token_projectors_only": train_token_projectors_only,
            }
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

        else:
            patience_counter += 1
            logging.info(f"No improvement. Patience: {patience_counter}/{patience}")


        # If no improvement for 'patience' epochs: stop training early to avoid overfitting/wasted compute
        if patience_counter >= patience:
            logging.info("Early stopping: Validation loss did not improve.")
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("best_val_loss_epoch", epoch)
            mlflow.log_metric("early_stopped_epoch", epoch + 1)

            break
        
    logging.info("\n" + "=" * 60)
    logging.info("Training complete! Testing...")
    logging.info("=" * 60)

    # -------------- FINAL EVALUATION ON TEST DATASET -------------------------------------------
    #df_test = df_test.iloc[0:1280]
    test_dataset = dataset_cls(
        df_test, tokenizer, num_virtual_tokens=num_virtual_tokens
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )

    # Load the best saved model state (from early stopping)
    if not train_token_projectors_only:
        model.load_state_dict(torch.load(os.path.join(output_dir, model_filename)))
    model = model.to(device)  # Ensure model is on the correct device

    # Load the best saved projector states from disk
    dna_projector.load_state_dict(
        torch.load(
            os.path.join(output_dir, "dna_projector.pt")
        )
    )
    dna_projector = dna_projector.to(device)  # Ensure on device

    protein_projector.load_state_dict(
        torch.load(
            os.path.join(output_dir, projector_filename)
        )
    )
    protein_projector = protein_projector.to(device)

    # -------------------------- EVALUATION -----------------------------------------------------
    # Evaluate best loss model on the test set (same fusion procedure)
    # Log final metrics
    test_loss = evaluate_model(
        model,
        dna_projector,
        protein_projector,
        test_loader,
        device,
        scale_virtual_embeddings,
        protein_batch_key=batch_key,
    )
    logging.info(f"Test loss: {test_loss:.4f}")
    mlflow.log_metric("test_loss", test_loss)

    # ======================== FINAL SUMMARY ========================
    logging.info("\n" + "=" * 60)
    logging.info("Trainig and testing complete!")
    logging.info(f"  Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Test loss: {test_loss:.4f}")
    logging.info(f"  Models saved to: {output_dir}")
    logging.info("=" * 60)

    # End MLflow run
    mlflow.end_run()
