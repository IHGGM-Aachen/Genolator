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


def test_padding_is_masked_out(pst_dataset):
    """No loss on padding, apart from the single stop token."""
    sample = pst_dataset[0]
    padding = sample["labels"][sample["attention_mask"] == 0]
    assert (padding[1:] == -100).all()


def test_one_eos_label_survives_in_padding(pst_dataset, tokenizer):
    """The first pad position stays supervised, otherwise nothing teaches the model to stop."""
    sample = pst_dataset[0]
    padding = sample["labels"][sample["attention_mask"] == 0]
    live = padding[padding != -100]
    assert live.tolist() == [tokenizer.eos_token_id]


def test_supervised_positions_are_response_plus_stop(pst_dataset, synthetic_rows, tokenizer):
    """Loss covers the response tokens plus the stop token, nothing else."""
    for idx in range(len(synthetic_rows)):
        sample = pst_dataset[idx]
        prompt_len = len(tokenizer(synthetic_rows.iloc[idx]["prompt"],
                                   truncation=True, max_length=MAX_LENGTH).input_ids)
        attended = int(sample["attention_mask"].sum())
        supervised = int((sample["labels"] != -100).sum())
        assert supervised == attended - prompt_len + 1, f"row {idx}"
        assert supervised > 0, f"row {idx} has no supervised position"


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
