from torch.nn import Module
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Any
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from utils.token_projector import GenomicVirtualTokenProjector
from utils.columns import DNA_EMBEDDING_COLUMN
import logging


# ------------------------------------ Utility Functions -----------------------------------------
def _is_invalid_emb(x: Any) -> bool:
    """
    Return True when an embedding is missing or unusable.

    Samples may lack a modality, so every embedding is checked before it reaches a
    projector; invalid ones are routed to the projector's learned null-embedding
    pathway instead of being fed forward.

    Args:
        x: value to check (None, NaN float, list, np.ndarray or torch.Tensor).

    Returns:
        bool: True if x is None, empty, or contains NaN. False otherwise.
    """
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, torch.Tensor):
        if x.numel() == 0 or torch.isnan(x).any():
            return True
    if isinstance(x, np.ndarray):
        if x.size == 0 or np.isnan(x).any():
            return True
    if isinstance(x, list):
        if len(x) == 0 or all(isinstance(xi, float) and np.isnan(xi) for xi in x):
            return True
    return False


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate multimodal QA samples into a batch for the PyTorch DataLoader.

    Token fields are stacked into tensors. Embedding fields are converted to float32
    individually and kept as lists rather than stacked, because a sample may be
    missing a modality and has to stay None instead of being padded; the projectors
    substitute their null embedding for those entries. Metadata passes through
    unchanged.

    Args:
        batch: list of sample dicts from GenomicQADatasetESM or GenomicQADatasetPST.

    Returns:
        Dict[str, Any]:
            {
                "input_ids": Tensor,        # [B, seq_len]
                "attention_mask": Tensor,   # [B, seq_len]
                "labels": Tensor,           # [B, seq_len]
                "nucleotide_emb": List[Optional[torch.Tensor]],
                "aa_emb": List[Optional[torch.Tensor]],
                "structure_emb": List[Optional[torch.Tensor]],
                "kind": List[str],          # question type per sample
                "response": List[str],      # ground truth answers
                "prompt": List[str],        # prompts
            }
    """
    # --- Stack token fields into batched tensors [batch_size, seq_len/feature_dim] ---
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    # --- Gather all embedding values as lists (as they may have different shapes/types) ---
    nucleotide_embs = [item.get("nucleotide_emb", None) for item in batch]
    aa_embs = [item.get("aa_emb", None) for item in batch]
    structure_embs = [item.get("structure_emb", None) for item in batch]

    # --- Handle nucleotide embeddings ---
    # Each element is either a numpy array, PyTorch Tensor, or None; convert to tensor if valid, else keep None.
    out_nuc = []
    for nuc in nucleotide_embs:
        if _is_invalid_emb(nuc):  # Check for None, NaNs, or empty
            out_nuc.append(None)
        elif isinstance(nuc, np.ndarray):
            out_nuc.append(torch.tensor(nuc, dtype=torch.float32))
        elif isinstance(nuc, torch.Tensor):
            out_nuc.append(nuc.float())
        else:
            out_nuc.append(
                torch.tensor(nuc, dtype=torch.float32)
            )  # Covers list of floats, etc.

    # --- Handle amino acid embeddings (same logic as nucleotide) ---
    out_aa = []
    for aa in aa_embs:
        if _is_invalid_emb(aa):
            out_aa.append(None)
        elif isinstance(aa, np.ndarray):
            out_aa.append(torch.tensor(aa, dtype=torch.float32))
        elif isinstance(aa, torch.Tensor):
            out_aa.append(aa.float())
        else:
            out_aa.append(torch.tensor(aa, dtype=torch.float32))

    # --- Handle structural embeddings (same logic as others) ---
    out_struct = []
    for st in structure_embs:
        if _is_invalid_emb(st):
            out_struct.append(None)
        elif isinstance(st, np.ndarray):
            out_struct.append(torch.tensor(st, dtype=torch.float32))
        elif isinstance(st, torch.Tensor):
            out_struct.append(st.float())
        else:
            out_struct.append(torch.tensor(st, dtype=torch.float32))

    # --- Gather metadata fields (kind, response, prompt) for GRPO training ---
    kinds = [item.get("kind", "unknown") for item in batch]
    responses = [item.get("response", "") for item in batch]
    prompts = [item.get("prompt", "") for item in batch]

    # --- Return dict of batch tensors (for tokens) and lists (for multimodal/projector input) ---
    # The lists are left as-is so projector modules can process each example individually,
    # including handling None values for missing modalities.
    return {
        "input_ids": input_ids,  # [B, seq_len]
        "attention_mask": attention_mask,  # [B, seq_len]
        "labels": labels,  # [B, seq_len]
        "nucleotide_emb": out_nuc,  # list of [emb_dim] or None, length = batch size
        "aa_emb": out_aa,  # list of [emb_dim] or None
        "structure_emb": out_struct,  # list of [emb_dim] or None
        "kind": kinds,  # list of task kinds (for GRPO)
        "response": responses,  # list of ground truth responses (for GRPO)
        "prompt": prompts,  # list of prompts (for GRPO generation)
    }


def evaluate_model(
    model: Module,
    dna_projector: Module,
    protein_projector: Module,
    loader: DataLoader,
    device: torch.device,
    scale_virtual_embeddings: bool = False,
    *,
    protein_batch_key: str,
) -> float:
    """
    Compute the mean loss of the multimodal model over a dataset.

    Mirrors the training forward pass: both projectors turn a sample's DNA and protein
    embeddings into virtual tokens, which are prepended to the text token embeddings.
    Labels are padded with -100 across the virtual positions, so the loss covers the
    answer tokens only.

    The model and both projectors are switched to eval mode for the pass and back to
    train mode before returning, so this is safe to call from inside the training loop.

    Args:
        model: LLaMA language model (PEFT/LoRA-wrapped).
        dna_projector: projector mapping DNA embeddings to virtual tokens.
        protein_projector: projector mapping protein embeddings to virtual tokens.
        loader: DataLoader yielding batches built by custom_collate.
        device: torch device to run on.
        scale_virtual_embeddings: rescale the virtual tokens to the mean L2 norm of
            the batch's text token embeddings.
        protein_batch_key: batch key holding the protein embeddings --
            "structure_emb" for PST, "aa_emb" for ESM-2.

    Returns:
        float: mean loss over the loader, or 0 if it yielded no batches.
    """
    # Switch all modules to eval mode (turn off dropout, etc.)
    model.eval()
    dna_projector.eval()
    protein_projector.eval()

    eval_loss = 0
    num_batches = 0

    # No grad for evaluation
    with torch.no_grad():
        for batch in loader:
            # Move token tensors to appropriate device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            dna_emb_list = batch[
                "nucleotide_emb"
            ]  # List of (emb_dim,); can include None
            protein_emb_list = batch[protein_batch_key]

            # --- Project dna embeddings ---
            vtoken_dna_embeds = []
            for emb in dna_emb_list:
                if emb is None:
                    vtoken_dna_embeds.append(
                        dna_projector().to(device)
                    )  # Null fallback
                else:
                    vtoken_dna_embeds.append(
                        dna_projector(emb.unsqueeze(0).to(device))
                    )  # Shape [1, N, H]
            vtoken_dna_embeds = torch.cat(
                vtoken_dna_embeds, dim=0
            )  # Shape [B, N, H]

            # --- Project amino acid embeddings ---
            vtoken_protein_embeds = []
            for emb in protein_emb_list:
                if emb is None:
                    vtoken_protein_embeds.append(protein_projector().to(device))
                else:
                    vtoken_protein_embeds.append(protein_projector(emb.unsqueeze(0).to(device)))
            vtoken_protein_embeds = torch.cat(vtoken_protein_embeds, dim=0)

            # --- Standard LM: get text (prompt + response) token embeddings ---
            text_embeds = model.get_input_embeddings()(
                input_ids
            )  # [B, seq_len, hidden]

            # --- Scale virtual token embeddings to match text embedding norms ---
            if scale_virtual_embeddings:
                text_norm = text_embeds.norm(dim=-1, keepdim=True).mean()
                dna_norm = vtoken_dna_embeds.norm(dim=-1, keepdim=True).mean()
                protein_norm = vtoken_protein_embeds.norm(dim=-1, keepdim=True).mean()
                vtoken_dna_embeds = vtoken_dna_embeds * (text_norm / (dna_norm + 1e-8))
                vtoken_protein_embeds = vtoken_protein_embeds * (text_norm / (protein_norm + 1e-8))

            # --- Concatenate: virtual tokens (all types) precede the text tokens ---
            inputs_embeds = torch.cat(
                [
                    vtoken_dna_embeds,
                    vtoken_protein_embeds,
                    text_embeds,
                ],
                dim=1,
            )
            # [B, num_virtuals_total + seq_len, hidden]

            # --- Prepare extended label and attention mask for the added virtual tokens ---
            num_all_virtuals = (
                vtoken_dna_embeds.shape[1]
                + vtoken_protein_embeds.shape[1]
            )
            virtual_pad = torch.full(
                (labels.size(0), num_all_virtuals),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )  # -100 means ignore in loss
            label_evaluation = torch.cat([virtual_pad, labels], dim=1)

            virtual_attn = torch.ones(
                (labels.size(0), num_all_virtuals),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask_evaluation = torch.cat([virtual_attn, attention_mask], dim=1)

            # --- Final sanity check on all tensor shapes ---
            assert (
                inputs_embeds.shape[1]
                == label_evaluation.shape[1]
                == attention_mask_evaluation.shape[1]
            ), "Shape mismatch in evaluation!"

            # --- Forward pass with all modalities and compute (masked) loss ---
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_evaluation,
                labels=label_evaluation,
            )
            eval_loss += outputs.loss.item()
            num_batches += 1

    # Switch back to train mode (important for correct BN/dropout when leaving this function)
    model.train()
    dna_projector.train()
    protein_projector.train()

    # Return mean loss
    return eval_loss / num_batches if num_batches > 0 else 0


def get_projected_embedding(
    emb: Any, projector: GenomicVirtualTokenProjector, device: torch.device
) -> torch.Tensor:
    """
    Project an embedding into virtual tokens, falling back to the null embedding.

    Args:
        emb: input embedding (may be missing or invalid).
        projector: projector module to apply.
        device: torch device for the returned tensor.

    Returns:
        torch.Tensor: projected virtual token tensor.
    """
    if _is_invalid_emb(emb):
        return projector().to(device)
    else:
        return projector(emb.unsqueeze(0).to(device))


def run_and_save_inference(
    df_subset: pd.DataFrame,
    llama_tokenizer: PreTrainedTokenizer,
    model: torch.nn.Module,
    dna_projector: GenomicVirtualTokenProjector,
    protein_projector: GenomicVirtualTokenProjector,
    device: torch.device,
    output_path: str,
    max_new_tokens: int = 500,
    collect_attention: bool = True,
    scale_virtual_embeddings: bool = False,
    *,
    protein_embedding_column: str,
    protein_attn_column: str,
) -> None:
    """
    Generate answers for a subset of samples and save the results to a pickle.

    For each row:
        1. Projects the cDNA and protein embeddings into virtual tokens.
        2. Prepends them to the prompt's text token embeddings.
        3. Generates greedily (argmax) with KV caching until an EOS token or
           max_new_tokens.
        4. Optionally records where the generated tokens put their attention.

    The attention traces are last-layer attention mass, summed over each input region
    and averaged across generation steps: the cDNA virtual tokens, the protein virtual
    tokens, the prompt text tokens, and the tokens generated so far. `hidden_states`
    is the mean last-layer hidden state of the generated positions.

    Args:
        df_subset: samples to run, with columns:
            - the DNA embedding column (utils/columns.DNA_EMBEDDING_COLUMN)
            - the protein embedding column named by protein_embedding_column
            - 'prompt': question to answer
            - 'response': ground truth answer, carried through for comparison
            - 'gene_name', 'go_aspect', 'kind': metadata, carried into the results.
              'seq_label' is carried as well when the input provides it.
        llama_tokenizer: tokenizer for the LLaMA model.
        model: LLaMA language model (PEFT/LoRA-wrapped).
        dna_projector: projector mapping cDNA embeddings to virtual tokens.
        protein_projector: projector mapping protein embeddings to virtual tokens.
        device: torch device to run on.
        output_path: pickle path for the results table.
        max_new_tokens: maximum tokens to generate per sample.
        collect_attention: record the attention traces. Needs an attention
            implementation that returns weights (the entrypoint sets "eager").
        scale_virtual_embeddings: rescale the virtual tokens to the mean L2 norm of
            the prompt's text token embeddings.
        protein_embedding_column: DataFrame column holding the protein embedding;
            PST_EMBEDDING_COLUMN or ESM_EMBEDDING_COLUMN from utils/columns.py.
        protein_attn_column: results column for the protein attention trace --
            "pst_attn" or "esm_attn".

    Returns:
        None. Writes a table to output_path with the metadata columns present in
        df_subset (gene_name, go_aspect, prompt, response, kind and, if present,
        seq_label) followed by generated_text, hidden_states, dna_attn,
        <protein_attn_column>, prompt_attn and output_attn.
    """
    generated_texts = []
    mean_hidden_states = []
    dna_attn_list = []
    protein_attn_list = []
    prompt_attn_list = []
    output_attn_list = []

    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        with torch.no_grad():
            # 1. Prepare and project genomic embeddings
            dna_emb = torch.as_tensor(row[DNA_EMBEDDING_COLUMN], dtype=torch.float32, device=device)
            protein_emb = torch.as_tensor(row[protein_embedding_column], dtype=torch.float32, device=device)

            vtoken_dna_embeds = get_projected_embedding(dna_emb, dna_projector, device)
            vtoken_protein_embeds = get_projected_embedding(protein_emb, protein_projector, device)

            # 2. Tokenize prompt and get text embeddings
            input_ids = llama_tokenizer(
                row["prompt"], return_tensors="pt"
            ).input_ids.to(device)
            text_embeds = model.get_input_embeddings()(input_ids)

            # 3. Scale virtual token embeddings to match text embedding norms
            if scale_virtual_embeddings:
                text_norm = text_embeds.norm(dim=-1, keepdim=True).mean()
                dna_norm = vtoken_dna_embeds.norm(dim=-1, keepdim=True).mean()
                protein_norm = vtoken_protein_embeds.norm(dim=-1, keepdim=True).mean()
                vtoken_dna_embeds = vtoken_dna_embeds * (text_norm / (dna_norm + 1e-8))
                vtoken_protein_embeds = vtoken_protein_embeds * (text_norm / (protein_norm + 1e-8))

            # 5. Multimodal fusion
            inputs_embeds = torch.cat([vtoken_dna_embeds, vtoken_protein_embeds, text_embeds], dim=1)

            # Token counts for attention analysis
            n_dna = vtoken_dna_embeds.shape[1]
            n_protein = vtoken_protein_embeds.shape[1]
            n_prompt = input_ids.shape[1]
            vtoken_count = n_dna + n_protein

            # 5. Initial attention mask
            attention_mask = torch.ones(
                (1, vtoken_count + n_prompt), dtype=torch.long, device=device
            )

            # 6. Autoregressive generation with KV caching
            generated_ids = []
            hidden_states_list = []
            attention_scores = []

            # First forward pass with full inputs_embeds to initialize KV cache
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                output_attentions=collect_attention,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values

            for _ in range(max_new_tokens):
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1)

                # Extract hidden state for this step
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                hidden_states_list.append(last_hidden.cpu().numpy())

                # Stop if EOS
                if next_token_id.item() == llama_tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id.item())

                # Collect attention weights if requested
                if collect_attention and outputs.attentions is not None:
                    last_attn = outputs.attentions[-1]
                    attn_weights = last_attn[0, :, -1, :].mean(0).cpu().numpy()

                    # Extract attention to input positions only (DNA + protein + prompt)
                    total_input_len = vtoken_count + n_prompt
                    attn_to_inputs = attn_weights[:total_input_len]

                    # Normalize to get relative attention within input tokens
                    # input_attn_sum = attn_to_inputs.sum()
                    # if input_attn_sum > 0:
                    #     attn_to_inputs = attn_to_inputs / input_attn_sum

                    attn_dna = attn_to_inputs[:n_dna].sum()
                    attn_protein = attn_to_inputs[n_dna:n_dna + n_protein].sum()
                    attn_prompt = attn_to_inputs[vtoken_count:].sum()
                    attn_output = attn_weights[total_input_len:].sum()  # Absolute attention to outputs

                    attention_scores.append({
                        "dna": attn_dna, 'protein': attn_protein,
                        "prompt": attn_prompt, "output": attn_output
                    })

                # Prepare next token embedding and extend attention mask
                next_token_embed = model.get_input_embeddings()(next_token_id.unsqueeze(0))
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=device)
                ], dim=1)

                # Forward pass with KV cache (only process new token)
                outputs = model(
                    inputs_embeds=next_token_embed,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=collect_attention,
                    output_hidden_states=True,
                )
                past_key_values = outputs.past_key_values

            # Aggregate results for this sample
            generated_texts.append(llama_tokenizer.decode(generated_ids, skip_special_tokens=True))

            if hidden_states_list:
                hidden_states_array = np.concatenate(hidden_states_list, axis=0)
                mean_hidden_states.append(np.mean(hidden_states_array, axis=0))
            else:
                mean_hidden_states.append(np.zeros(model.config.hidden_size))

            if collect_attention and attention_scores:
                dna_attn_list.append(np.mean([s['dna'] for s in attention_scores]))
                protein_attn_list.append(np.mean([s['protein'] for s in attention_scores]))
                prompt_attn_list.append(np.mean([s['prompt'] for s in attention_scores]))
                output_attn_list.append(np.mean([s['output'] for s in attention_scores]))
            else:
                dna_attn_list.append(0.0)
                protein_attn_list.append(0.0)
                prompt_attn_list.append(0.0)
                output_attn_list.append(0.0)

            # Clear KV cache to free memory
            del past_key_values

    # Add results to DataFrame
    df_subset["generated_text"] = generated_texts
    df_subset["hidden_states"] = mean_hidden_states
    df_subset["dna_attn"] = dna_attn_list
    df_subset[protein_attn_column] = protein_attn_list
    df_subset["prompt_attn"] = prompt_attn_list
    df_subset["output_attn"] = output_attn_list

    # Save to pickle. Metadata columns are written only when the input actually
    # provides them, so a split without an optional column still works.
    metadata_columns = [
        column
        for column in ("gene_name", "go_aspect", "prompt", "response", "kind", "seq_label")
        if column in df_subset.columns
    ]
    result_columns = [
        "generated_text", "hidden_states", "dna_attn", protein_attn_column,
        "prompt_attn", "output_attn",
    ]
    df_subset[metadata_columns + result_columns].to_pickle(output_path)
