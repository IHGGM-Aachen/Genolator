from typing import ClassVar

from torch.utils.data import Dataset

from utils.columns import (
    DNA_BATCH_KEY,
    DNA_EMBEDDING_COLUMN,
    ESM_BATCH_KEY,
    ESM_EMBEDDING_COLUMN,
    PST_BATCH_KEY,
    PST_EMBEDDING_COLUMN,
)

# ----------------------------- CUSTOM DATASET ----------------------------------------------------------


class GenomicQADatasetBase(Dataset):
    """
    PyTorch Dataset for Genomics QA.

    Handles the parts that are identical across modality combinations: building the
    LM-style input from prompt + response, masking the prompt out of the labels, and
    carrying the per-sample metadata through to the collate function.

    Subclasses only declare EMBEDDING_COLUMNS, which maps the key consumed downstream
    (in the training/inference loops and in `custom_collate`) to the DataFrame column
    holding the corresponding precomputed embedding.

    Embeddings are returned untouched; `custom_collate` converts them to tensors and
    substitutes None for missing or NaN entries, so a projector's learnable null
    embedding is used for samples where a modality is unavailable.
    """

    EMBEDDING_COLUMNS: ClassVar[dict[str, str]] = {}

    def __init__(self, data, tokenizer, max_length=256, num_virtual_tokens=8):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_virtual_tokens = num_virtual_tokens

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        prompt = item["prompt"]
        response = item["response"]

        # Concatenate NL prompt + target response for LM-style input
        input_text = prompt + " " + response
        enc = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc.input_ids.squeeze(0)

        # Mask out prompt tokens in label for LM loss (no loss on prompt part)
        prompt_enc = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        prompt_len = prompt_enc.input_ids.shape[-1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Use -100 to ignore in LM loss

        # Return all fields -- token IDs, metadata and embedding features
        sample = {
            "input_ids": input_ids,
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": labels,
            "kind": item.get("kind", "unknown"),
            "response": response,
            "prompt": prompt,
        }

        for batch_key, column in self.EMBEDDING_COLUMNS.items():
            sample[batch_key] = item[column]

        return sample

    def __len__(self):
        return len(self.data)


class GenomicQADatasetESM(GenomicQADatasetBase):
    """
    Genomics QA over nucleotide (Evo2) and amino acid (ESM-2) embeddings.
    Selected by train_genolator.py --embedding_type esm.
    """

    EMBEDDING_COLUMNS: ClassVar[dict[str, str]] = {
        DNA_BATCH_KEY: DNA_EMBEDDING_COLUMN,
        ESM_BATCH_KEY: ESM_EMBEDDING_COLUMN,
    }


class GenomicQADatasetPST(GenomicQADatasetBase):
    """
    Genomics QA over nucleotide (Evo2) and protein structure (PST) embeddings.
    Selected by train_genolator.py --embedding_type pst.
    """

    EMBEDDING_COLUMNS: ClassVar[dict[str, str]] = {
        DNA_BATCH_KEY: DNA_EMBEDDING_COLUMN,
        PST_BATCH_KEY: PST_EMBEDDING_COLUMN,
    }
