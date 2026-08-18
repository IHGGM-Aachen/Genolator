"""
Split loading -- the published Hugging Face dataset by default, local files on request.

Both entrypoints call `load_split`, which resolves its `source` argument in one of
two ways:

* an existing local path -- a single file (`.parquet` or a pickled DataFrame), or a
  directory searched for `<split>.parquet`, `data/<split>.parquet` and `<split>.pkl`;
* anything else -- a Hugging Face dataset repo id. The split's parquet file is
  downloaded through `huggingface_hub` and cached, so repeated runs do not refetch.

Parquet reads are column-pruned: only the columns a run actually touches are
decoded. The raw sequence columns (`dna_seq`, `cdna_seq`, `aa_seq`) are the bulk of
the dataset on disk and are never read by training or inference, so pruning them
matters -- see the size figures in README.md. Pickled DataFrames cannot be pruned
and are read whole.

Nothing here depends on the `datasets` library; `huggingface_hub` (already needed
for the gated Llama-3 weights) plus pandas/pyarrow are enough.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .columns import DNA_EMBEDDING_COLUMN

# The dataset released alongside the paper. Splits are already gene-disjoint, so
# there is no splitting step in this code.
DEFAULT_DATASET = "CHGGM-Aachen/genolator-v1-qa"

SPLITS = ("train", "validation", "test")

# Where each split lives inside the dataset repo.
REPO_FILES = {split: f"data/{split}.parquet" for split in SPLITS}

# Metadata columns read during training. `group` is the epoch rotation subset and
# exists in the train split only.
TRAIN_METADATA_COLUMNS = ["prompt", "response", "kind"]
TRAIN_GROUP_COLUMN = "group"

# Metadata columns carried into the inference results table.
INFERENCE_METADATA_COLUMNS = ["gene_name", "go_aspect", "prompt", "response", "kind"]

_PICKLE_SUFFIXES = (".pkl", ".pickle")


def training_columns(protein_embedding_column: str, *, with_group: bool) -> list[str]:
    """Columns training reads: text, kind, both embeddings, and `group` on train."""
    columns = [*TRAIN_METADATA_COLUMNS, DNA_EMBEDDING_COLUMN, protein_embedding_column]
    if with_group:
        columns.append(TRAIN_GROUP_COLUMN)
    return columns


def inference_columns(protein_embedding_column: str) -> list[str]:
    """Columns inference reads: results metadata plus both embeddings."""
    return [*INFERENCE_METADATA_COLUMNS, DNA_EMBEDDING_COLUMN, protein_embedding_column]


def _local_split_file(directory: str, split: str) -> str | None:
    """First recognised file for `split` inside a local directory, if any."""
    candidates = [
        os.path.join(directory, f"{split}.parquet"),
        os.path.join(directory, "data", f"{split}.parquet"),
        os.path.join(directory, f"{split}.pkl"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _prune(path: str, columns: list[str] | None) -> list[str] | None:
    """Requested columns narrowed to those the parquet file actually has."""
    if columns is None:
        return None
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [column for column in columns if column not in available]
    if missing:
        logging.warning(
            "Columns absent from %s and not read: %s", path, ", ".join(missing)
        )
    return [column for column in columns if column in available]


def _read_file(path: str, columns: list[str] | None) -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=_prune(path, columns))
    if suffix in _PICKLE_SUFFIXES:
        # A pickled DataFrame is materialised whole; `columns` cannot prune it.
        return pd.read_pickle(path)
    raise ValueError(
        f"Unsupported dataset file '{path}': expected .parquet or a pickled "
        f"DataFrame ({', '.join(_PICKLE_SUFFIXES)})."
    )


def load_split(
    source: str,
    split: str,
    columns: list[str] | None = None,
    token: str | None = None,
    revision: str | None = None,
) -> pd.DataFrame:
    """
    Load one split as a DataFrame.

    Args:
        source: a Hugging Face dataset repo id (e.g. the DEFAULT_DATASET above), or
            a local path to a split file or to a directory holding the splits.
        split: 'train', 'validation' or 'test'. Selects the file when `source` is a
            repo id or a directory; ignored when `source` points at a single file.
        columns: columns to read. Only applied to parquet; columns the file does not
            have are skipped with a warning. None reads everything.
        token: Hugging Face token, for a dataset repo that is private or gated.
        revision: dataset repo revision (branch, tag or commit sha). None takes the
            default branch; pin a commit to make a run exactly reproducible.

    Returns:
        The split as a pandas DataFrame. Embedding columns come back as 1-D float32
        numpy arrays, which is what `custom_collate` expects.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}': expected one of {', '.join(SPLITS)}.")

    if os.path.isfile(source):
        logging.info("Loading %s split from file %s", split, source)
        return _read_file(source, columns)

    if os.path.isdir(source):
        path = _local_split_file(source, split)
        if path is None:
            raise FileNotFoundError(
                f"No file for split '{split}' in directory '{source}'. Expected "
                f"'{split}.parquet', 'data/{split}.parquet' or '{split}.pkl'."
            )
        logging.info("Loading %s split from file %s", split, path)
        return _read_file(path, columns)

    logging.info(
        "Loading %s split from Hugging Face dataset %s%s",
        split,
        source,
        f" (revision {revision})" if revision else "",
    )
    path = hf_hub_download(
        repo_id=source,
        filename=REPO_FILES[split],
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    return _read_file(path, columns)


def subset_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    max_samples: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cap each split to roughly `max_samples` rows. Intended for smoke tests.

    `max_samples=None` returns the frames untouched, which is the normal path for a
    real run.

    Validation and test are truncated with a plain `head`. The training split is not:
    the epoch loop rotates through the values of `group` (`epoch % 3`), so a globally
    truncated frame that happens to be ordered by group would leave the later groups
    empty and hand those epochs an empty DataLoader. The cap is therefore spread
    evenly across groups instead, which keeps every epoch of the rotation non-empty.
    The cost is that the training split can come back slightly larger than
    `max_samples` when the number of groups does not divide it evenly.

    A training frame without a `group` column -- possible via --train_path -- is
    capped with a plain `head`, since there is no rotation to preserve.
    """
    if max_samples is None:
        return df_train, df_val, df_test
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")

    if TRAIN_GROUP_COLUMN in df_train.columns:
        n_groups = max(1, df_train[TRAIN_GROUP_COLUMN].nunique())
        per_group = -(-max_samples // n_groups)  # ceiling division
        df_train = df_train.groupby(TRAIN_GROUP_COLUMN, group_keys=False).head(per_group)
    else:
        df_train = df_train.head(max_samples)

    df_val = df_val.head(max_samples)
    df_test = df_test.head(max_samples)

    logging.info(
        "max_samples=%d: capped to %d train / %d validation / %d test rows",
        max_samples, len(df_train), len(df_val), len(df_test),
    )
    return df_train, df_val, df_test
