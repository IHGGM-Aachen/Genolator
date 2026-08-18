"""The argument surface of train_genolator.py."""

from __future__ import annotations

import pytest

import train_genolator as tg

BASE = ["train_genolator.py", "--embedding_type", "pst",
        "--output_dir", "/tmp/out", "--hf_token", "token"]


def parse(monkeypatch, *extra):
    monkeypatch.setattr("sys.argv", BASE + list(extra))
    return tg.parse_args()


def test_defaults(monkeypatch):
    args = parse(monkeypatch)
    assert args.max_samples is None          # full splits unless asked otherwise
    assert args.learning_rate == 5e-5
    assert args.projector_learning_rate is None
    assert args.batch_size == 8
    assert args.num_epochs == 10
    assert args.patience == 3
    assert args.lora_r == 8
    assert args.num_virtual_tokens == 8


def test_max_samples_parses_as_int(monkeypatch):
    assert parse(monkeypatch, "--max_samples", "512").max_samples == 512


def test_learning_rates_are_independent(monkeypatch):
    args = parse(monkeypatch, "--learning_rate", "1e-4",
                 "--projector_learning_rate", "3e-4")
    assert (args.learning_rate, args.projector_learning_rate) == (1e-4, 3e-4)


@pytest.mark.parametrize("embedding_type", ["pst", "esm"])
def test_both_modalities_accepted(monkeypatch, embedding_type):
    monkeypatch.setattr("sys.argv", BASE[:2] + [embedding_type] + BASE[3:])
    assert tg.parse_args().embedding_type == embedding_type


def test_unknown_modality_rejected(monkeypatch):
    monkeypatch.setattr("sys.argv", BASE[:2] + ["bogus"] + BASE[3:])
    with pytest.raises(SystemExit):
        tg.parse_args()


@pytest.mark.parametrize("missing", ["--embedding_type", "--output_dir", "--hf_token"])
def test_required_arguments(monkeypatch, missing):
    argv = list(BASE)
    idx = argv.index(missing)
    del argv[idx:idx + 2]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        tg.parse_args()


def test_modalities_table_matches_choices():
    """Every modality in MODALITIES carries the keys the entrypoint reads."""
    for name, spec in tg.MODALITIES.items():
        assert {"label", "embedding_column", "model_filename"} <= set(spec), name
