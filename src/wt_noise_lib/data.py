"""Data loading and the grouped train/val/test split.

The raw file is a tab-separated text file with no header and 1,503 rows of six
columns (five inputs + the SPL target). See :mod:`wt_noise_lib.config`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Airfoil Self-Noise dataset into a typed, named DataFrame.

    Parameters
    ----------
    path
        Location of ``airfoil_self_noise.dat``. Defaults to the repo copy at
        :data:`wt_noise_lib.config.DATA_PATH`.

    Returns
    -------
    pandas.DataFrame
        Shape ``(1503, 6)`` with the columns of :data:`config.COLUMNS`.
    """
    path = Path(path) if path is not None else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path!s}. Download it from {config.DATA_URL}"
        )
    df = pd.read_csv(path, sep="\t", header=None, names=config.COLUMNS)
    return df


def config_group_ids(df: pd.DataFrame) -> pd.Series:
    """Return a group id per row identifying its wind-tunnel configuration.

    Rows sharing the same ``CONFIG_COLUMNS`` values (they differ only in the
    swept frequency) belong to the same physical setup and must not be split
    across folds.
    """
    return df.groupby(config.CONFIG_COLUMNS, sort=False).ngroup()


def grouped_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (train, val, test) grouped by configuration.

    Whole configurations are kept together so the model is always evaluated on
    *unseen physical setups*, giving an honest generalization estimate. Returns
    three disjoint DataFrames (row order preserved within each).
    """
    groups = config_group_ids(df)

    # First carve off the test set by group.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss.split(df, groups=groups))
    df_trainval = df.iloc[train_val_idx]
    df_test = df.iloc[test_idx]

    # Then carve validation out of the remainder, again by group.
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tv_groups = groups.iloc[train_val_idx]
    train_idx, val_idx = next(gss_val.split(df_trainval, groups=tv_groups))
    df_train = df_trainval.iloc[train_idx]
    df_val = df_trainval.iloc[val_idx]

    return df_train, df_val, df_test
