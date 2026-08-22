"""Input guardrails that sit *upstream* of the regressor.

Layer 1 -- :func:`validate_physical`: rejects the physically impossible using
hard, spec-derived bounds. A neural net should never be asked to relearn the
laws of physics.

Layer 2 -- :class:`OODDetector`: flags inputs that are physically valid but lie
outside the training distribution, so an extrapolated prediction can be marked
low-confidence / abstain rather than trusted blindly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import config


# --------------------------------------------------------------------------- #
# Layer 1 -- physical validity
# --------------------------------------------------------------------------- #
def validate_physical(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Check each row against hard physical bounds.

    Returns a copy of ``df`` with two added columns:
    ``is_physically_valid`` (bool) and ``violation`` (str, empty if valid).
    """
    bounds = bounds if bounds is not None else config.PHYSICAL_BOUNDS
    out = df.copy()
    violations = pd.Series([""] * len(df), index=df.index)

    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        if lo is not None:
            bad = df[col] < lo
            violations.loc[bad & (violations == "")] = f"{col} < {lo}"
        if hi is not None:
            bad = df[col] > hi
            violations.loc[bad & (violations == "")] = f"{col} > {hi}"

    out["violation"] = violations
    out["is_physically_valid"] = violations == ""
    return out


# --------------------------------------------------------------------------- #
# Layer 2 -- out-of-distribution / low-confidence detection
# --------------------------------------------------------------------------- #
class OODDetector:
    """Flag inputs that fall outside the training feature distribution.

    Fit on the TRAINING features only. Combines a fast bounding-box check (are
    we outside the observed per-feature range?) with an IsolationForest novelty
    score for interior-but-sparse regions.
    """

    def __init__(self, contamination: float = 0.01, seed: int = config.RANDOM_SEED):
        self.contamination = contamination
        self.seed = seed
        self._scaler = StandardScaler()
        self._iforest = IsolationForest(
            contamination=contamination, random_state=seed, n_estimators=200
        )
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self.features_ = list(config.FEATURE_COLUMNS)

    def fit(self, X: pd.DataFrame) -> "OODDetector":
        X = X[self.features_]
        self._min = X.min().to_numpy()
        self._max = X.max().to_numpy()
        self._iforest.fit(self._scaler.fit_transform(X))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return a boolean Series: True where the row is out-of-distribution."""
        X = X[self.features_]
        outside_box = ((X.to_numpy() < self._min) | (X.to_numpy() > self._max)).any(axis=1)
        novel = self._iforest.predict(self._scaler.transform(X)) == -1
        return pd.Series(outside_box | novel, index=X.index, name="is_ood")
