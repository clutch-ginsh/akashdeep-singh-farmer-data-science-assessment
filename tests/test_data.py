"""Tests for data loading and the grouped split."""

import pandas as pd

import wt_noise_lib as wtai
from wt_noise_lib import config
from wt_noise_lib.data import grouped_split


def test_load_shape_and_columns():
    df = wtai.load_data()
    assert df.shape == (1503, 6)
    assert list(df.columns) == config.COLUMNS
    assert df.isna().sum().sum() == 0


def test_grouped_split_is_disjoint_and_grouped():
    df = wtai.load_data()
    train, val, test = grouped_split(df)

    # Partition covers every row exactly once.
    assert len(train) + len(val) + len(test) == len(df)
    idx = set(train.index) | set(val.index) | set(test.index)
    assert len(idx) == len(df)

    # No configuration leaks across folds.
    def cfgs(frame: pd.DataFrame) -> set:
        return set(map(tuple, frame[config.CONFIG_COLUMNS].to_numpy()))

    assert cfgs(train).isdisjoint(cfgs(test))
    assert cfgs(train).isdisjoint(cfgs(val))
    assert cfgs(val).isdisjoint(cfgs(test))
