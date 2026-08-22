"""Synthetic anomaly injection (Step 12) and residual-based detection (Step 13).

Injection models *instrument faults on the acoustic sensor*: the five input
features stay clean and only ``sound_pressure_db`` is corrupted, matching the
brief. The default fault is an additive dB spike at low / moderate / severe
magnitudes; the taxonomy is designed to be extended (saturation, dropout,
drift) as a robustness study.

Detection compares the measured SPL against the model's expected SPL and flags
large residuals. The threshold is calibrated on CLEAN residuals (train/val)
only -- never on the injected test set (see README decision log).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class InjectionResult:
    """Container for an injected dataset and its ground truth."""

    values: np.ndarray            # corrupted SPL values (dB)
    is_anomaly: np.ndarray        # bool labels, True where corrupted
    severity: np.ndarray          # str per row: "" or low/moderate/severe
    delta_db: np.ndarray          # signed dB offset applied (0 where clean)


def inject_spikes(
    y_db: np.ndarray,
    fraction: float = config.INJECTION_FRACTION,
    severity_db: dict[str, tuple[float, float]] | None = None,
    seed: int = config.RANDOM_SEED,
) -> InjectionResult:
    """Inject additive dB spikes into a copy of ``y_db``.

    A ``fraction`` of positions is chosen at random and split evenly across the
    severity levels. Each gets a signed offset drawn uniformly from that
    level's dB range (both signs -- faults push readings high or low).

    Parameters
    ----------
    y_db
        Clean SPL target values (test set), in dB.
    fraction
        Fraction of positions to corrupt.
    severity_db
        Mapping severity -> (min_db, max_db) magnitude. Defaults to
        :data:`config.SEVERITY_DB`.
    seed
        RNG seed for reproducibility.
    """
    severity_db = severity_db if severity_db is not None else config.SEVERITY_DB
    rng = np.random.default_rng(seed)

    y = np.asarray(y_db, dtype=float).copy()
    n = len(y)
    is_anomaly = np.zeros(n, dtype=bool)
    severity = np.array([""] * n, dtype=object)
    delta = np.zeros(n, dtype=float)

    n_anom = int(round(fraction * n))
    if n_anom == 0:
        return InjectionResult(y, is_anomaly, severity, delta)

    idx = rng.choice(n, size=n_anom, replace=False)
    levels = list(severity_db.keys())
    buckets = np.array_split(idx, len(levels))

    for level, bucket in zip(levels, buckets):
        lo, hi = severity_db[level]
        mag = rng.uniform(lo, hi, size=len(bucket))
        sign = rng.choice([-1.0, 1.0], size=len(bucket))
        d = sign * mag
        y[bucket] += d
        delta[bucket] = d
        is_anomaly[bucket] = True
        severity[bucket] = level

    return InjectionResult(y, is_anomaly, severity, delta)


class ResidualAnomalyDetector:
    """Flag anomalies from the regression residual measured - predicted.

    Calibrate the threshold on CLEAN residuals via a robust MAD-based z-score,
    then apply the frozen threshold to (possibly corrupted) test residuals.
    """

    def __init__(self, z_thresh: float = 3.5):
        self.z_thresh = z_thresh
        self.median_: float | None = None
        self.mad_: float | None = None

    def fit(self, clean_residuals: np.ndarray) -> "ResidualAnomalyDetector":
        """Calibrate on residuals from clean (train/val) predictions only."""
        r = np.asarray(clean_residuals, dtype=float)
        self.median_ = float(np.median(r))
        # 1.4826 scales MAD to be a consistent estimator of sigma for normal data.
        self.mad_ = float(1.4826 * np.median(np.abs(r - self.median_)) + 1e-12)
        return self

    def score(self, residuals: np.ndarray) -> np.ndarray:
        """Return the robust |z-score| anomaly score for each residual."""
        r = np.asarray(residuals, dtype=float)
        return np.abs((r - self.median_) / self.mad_)

    def predict(self, residuals: np.ndarray) -> np.ndarray:
        """Return a boolean array: True where the residual is anomalous."""
        return self.score(residuals) > self.z_thresh
