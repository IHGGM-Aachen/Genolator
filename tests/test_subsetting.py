"""utils.data.subset_splits -- the --max_samples cap."""

from __future__ import annotations

import pytest

from utils.data import TRAIN_GROUP_COLUMN, subset_splits


def test_none_is_a_no_op(grouped_splits):
    train, val, test = grouped_splits
    out = subset_splits(train, val, test, None)
    assert [len(df) for df in out] == [len(train), len(val), len(test)]


def test_val_and_test_are_capped_exactly(grouped_splits):
    _, val, test = subset_splits(*grouped_splits, 120)
    assert len(val) == 120
    assert len(test) == 120


def test_every_group_survives_the_cap(grouped_splits):
    """
    The regression this function exists for. The epoch loop rotates through groups
    (epoch % 3); a plain head() on a group-ordered frame leaves later groups empty
    and gives those epochs an empty DataLoader.
    """
    train, _, _ = subset_splits(*grouped_splits, 120)
    counts = train[TRAIN_GROUP_COLUMN].value_counts()
    assert set(counts.index) == {1, 2, 3}
    assert counts.min() > 0
    assert len(train) == pytest.approx(120, abs=3)


def test_cap_larger_than_split_returns_everything(grouped_splits):
    train, val, test = grouped_splits
    out = subset_splits(train, val, test, 10 ** 6)
    assert [len(df) for df in out] == [len(train), len(val), len(test)]


def test_train_without_group_column_falls_back_to_head(grouped_splits):
    train, val, test = grouped_splits
    ungrouped = train.drop(columns=[TRAIN_GROUP_COLUMN])
    out_train, _, _ = subset_splits(ungrouped, val, test, 50)
    assert len(out_train) == 50


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive(grouped_splits, bad):
    with pytest.raises(ValueError):
        subset_splits(*grouped_splits, bad)


def test_inputs_are_not_mutated(grouped_splits):
    train, val, test = grouped_splits
    before = (len(train), len(val), len(test))
    subset_splits(train, val, test, 30)
    assert (len(train), len(val), len(test)) == before
