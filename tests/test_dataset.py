"""
GenomicQADataset* -- tokenisation, label masking, and the collate contract.

Uses synthetic rows: this checks wiring, not data.
"""

from __future__ import annotations

import pytest
import torch

from utils.columns import DNA_BATCH_KEY, ESM_BATCH_KEY, PST_BATCH_KEY
from utils.qa_dataset import GenomicQADatasetESM, GenomicQADatasetPST
from utils.utils import custom_collate

MAX_LENGTH = 256


@pytest.fixture
def pst_dataset(synthetic_rows, tokenizer):
    return GenomicQADatasetPST(synthetic_rows, tokenizer, max_length=MAX_LENGTH)


def test_sample_shapes(pst_dataset):
    sample = pst_dataset[0]
    for key in ("input_ids", "labels", "attention_mask"):
        assert sample[key].shape == (MAX_LENGTH,), key


def test_prompt_tokens_are_masked_out(pst_dataset, synthetic_rows, tokenizer):
    """No loss on the prompt: those label positions must be -100."""
    prompt_len = len(tokenizer(synthetic_rows.iloc[0]["prompt"],
                               truncation=True, max_length=MAX_LENGTH).input_ids)
    labels = pst_dataset[0]["labels"]
    assert (labels[:prompt_len] == -100).all()
    assert (labels[prompt_len:] != -100).any(), "the response must survive masking"


def test_no_eos_inside_the_attended_region(pst_dataset, tokenizer):
    """The stop token only ever appears in the padding, never in the real text."""
    sample = pst_dataset[0]
    real = sample["input_ids"][sample["attention_mask"] == 1]
    assert not (real == tokenizer.eos_token_id).any()


def test_length_matches_frame(pst_dataset, synthetic_rows):
    assert len(pst_dataset) == len(synthetic_rows)


@pytest.mark.parametrize(
    "cls,protein_key",
    [(GenomicQADatasetPST, PST_BATCH_KEY), (GenomicQADatasetESM, ESM_BATCH_KEY)],
)
def test_collate_emits_expected_batch_keys(synthetic_rows, tokenizer, cls, protein_key):
    dataset = cls(synthetic_rows, tokenizer, max_length=MAX_LENGTH)
    batch = custom_collate([dataset[i] for i in range(len(dataset))])

    assert {"input_ids", "labels", "attention_mask"} <= set(batch)
    assert batch["input_ids"].shape == (len(dataset), MAX_LENGTH)
    assert batch["labels"].shape == batch["input_ids"].shape
    assert DNA_BATCH_KEY in batch and protein_key in batch
    assert len(batch[DNA_BATCH_KEY]) == len(dataset)


def test_missing_modality_becomes_none(synthetic_rows, tokenizer):
    """
    Row 3 has no protein embedding. custom_collate must hand the projector None so
    it can substitute its learnable null embedding, rather than passing NaN through.
    """
    dataset = GenomicQADatasetPST(synthetic_rows, tokenizer, max_length=MAX_LENGTH)
    batch = custom_collate([dataset[i] for i in range(len(dataset))])

    assert batch[PST_BATCH_KEY][3] is None
    present = [e for e in batch[PST_BATCH_KEY] if e is not None]
    assert len(present) == len(dataset) - 1
    assert all(torch.isfinite(torch.as_tensor(e)).all() for e in present)
