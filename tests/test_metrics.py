"""Tests for the regression and detection metric helpers."""

import numpy as np

from wt_noise_lib.metrics import detection_metrics, regression_metrics


def test_regression_metrics_perfect_prediction():
    y = np.array([100.0, 110.0, 120.0])
    m = regression_metrics(y, y)
    assert m["rmse_db"] == 0.0
    assert m["mae_db"] == 0.0
    assert m["r2"] == 1.0


def test_detection_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.95])
    y_pred = np.array([0, 0, 1, 1])
    m = detection_metrics(y_true, y_score, y_pred)
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
