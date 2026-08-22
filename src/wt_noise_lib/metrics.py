"""Evaluation metrics for the two questions the challenge poses.

Regression (is the DL model actually best?):
    * RMSE -- in dB; squares errors so it penalises the large deviations that
      matter most for severe acoustic faults. The headline regression number.
    * MAE  -- in dB; the typical error magnitude, robust to a few outliers.
    * R^2  -- fraction of SPL variance explained; unitless, comparable across models.

Detection (does anomaly detection work?):
    * Precision / Recall / F1 -- at the operating threshold.
    * PR-AUC -- primary detection metric: anomalies are rare, so the
      precision-recall trade-off is more informative than ROC.
    * ROC-AUC -- reported for completeness.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R^2 for SPL predictions (all in dB except R^2)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse_db": rmse,
        "mae_db": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def detection_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """Anomaly-detection metrics.

    Parameters
    ----------
    y_true
        Binary ground-truth labels (1 = injected anomaly).
    y_score
        Continuous anomaly score (e.g. absolute residual). Used for the AUCs.
    y_pred
        Optional binary predictions at the chosen threshold. If omitted, only
        the threshold-free AUC metrics are returned.
    """
    out: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
    if y_pred is not None:
        out.update(
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
        )
    return out
