"""
Shared fixtures.

`unsloth` is stubbed before anything imports `train_genolator`: it resolves only on
CUDA platforms, but it is imported at module scope in the entrypoint, so without a
stub none of the CLI tests could run on a laptop. Nothing here exercises unsloth
itself -- the model loading and training loop need a GPU and are out of scope for
this suite.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if "unsloth" not in sys.modules:
    _stub = types.ModuleType("unsloth")
    _stub.FastLanguageModel = object
    sys.modules["unsloth"] = _stub

LLAMA_MODEL = "ContactDoctor/Bio-Medical-Llama-3-8B"

# Embedding widths of the published dataset, per README.
DNA_DIM, ESM_DIM, PST_DIM = 4096, 2560, 1280


@pytest.fixture(scope="session")
def tokenizer():
    """
    The real Llama-3 tokenizer. Skipped when it cannot be reached: the weights are
    gated, so this needs a cached copy or HF_TOKEN plus network.
    """
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    except Exception as exc:  # gated, offline, or no token
        pytest.skip(f"Llama-3 tokenizer unavailable ({type(exc).__name__}); "
                    f"set HF_TOKEN or pre-cache {LLAMA_MODEL}")
    # train_genolator.py does exactly this before building the datasets.
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture
def synthetic_rows():
    """
    A DataFrame shaped like a split: real column names, random embeddings.

    Synthetic on purpose -- these tests check wiring, not data, and the published
    splits are ~8 GiB. Row 3 has a missing protein embedding so the projector's
    null-embedding path gets exercised.
    """
    from utils.columns import (
        DNA_EMBEDDING_COLUMN,
        ESM_EMBEDDING_COLUMN,
        PST_EMBEDDING_COLUMN,
    )

    rng = np.random.default_rng(0)
    n = 8
    rows = pd.DataFrame({
        "gene_name": [f"GENE{i}" for i in range(n)],
        "prompt": [f"Does GENE{i} participate in signal transduction?" for i in range(n)],
        "response": ["Yes, the evidence supports this function." for _ in range(n)],
        "kind": (["confirmation", "denial", "generic"] * n)[:n],
        DNA_EMBEDDING_COLUMN: [rng.normal(size=DNA_DIM).astype(np.float32) for _ in range(n)],
        ESM_EMBEDDING_COLUMN: [rng.normal(size=ESM_DIM).astype(np.float32) for _ in range(n)],
        PST_EMBEDDING_COLUMN: [rng.normal(size=PST_DIM).astype(np.float32) for _ in range(n)],
    })
    rows.at[3, PST_EMBEDDING_COLUMN] = None
    return rows


@pytest.fixture
def grouped_splits():
    """Train frame ordered by `group` -- the worst case for a naive global cap."""
    from utils.data import TRAIN_GROUP_COLUMN

    n = 900
    train = pd.DataFrame({
        "prompt": [f"q{i}" for i in range(n)],
        "response": [f"a{i}" for i in range(n)],
        "kind": ["confirmation"] * n,
        TRAIN_GROUP_COLUMN: np.repeat([1, 2, 3], n // 3),
    })
    other = train.drop(columns=[TRAIN_GROUP_COLUMN]).copy()
    return train, other.copy(), other.copy()
