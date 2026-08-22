"""Tests for synthetic anomaly injection and residual detection."""

import numpy as np

from wt_noise_lib.anomaly import ResidualAnomalyDetector, inject_spikes


def test_injection_touches_only_expected_fraction():
    y = np.full(1000, 125.0)
    res = inject_spikes(y, fraction=0.10, seed=0)
    assert res.is_anomaly.sum() == 100
    # Clean positions are untouched; corrupted positions changed.
    assert np.allclose(res.values[~res.is_anomaly], 125.0)
    assert np.all(res.values[res.is_anomaly] != 125.0)


def test_injection_is_reproducible():
    y = np.full(500, 120.0)
    a = inject_spikes(y, seed=7)
    b = inject_spikes(y, seed=7)
    assert np.array_equal(a.is_anomaly, b.is_anomaly)
    assert np.allclose(a.values, b.values)


def test_severity_magnitudes_increase():
    y = np.zeros(3000)
    res = inject_spikes(y, fraction=0.9, seed=1)
    mags = {lvl: np.abs(res.delta_db[res.severity == lvl]).mean()
            for lvl in ("low", "moderate", "severe")}
    assert mags["low"] < mags["moderate"] < mags["severe"]


def test_residual_detector_flags_large_residuals():
    rng = np.random.default_rng(0)
    clean = rng.normal(0.0, 1.0, size=2000)  # clean residuals ~ N(0,1)
    det = ResidualAnomalyDetector(z_thresh=3.5).fit(clean)

    test = np.array([0.0, 0.5, 20.0, -25.0])  # last two are gross outliers
    flags = det.predict(test)
    assert not flags[0] and not flags[1]
    assert flags[2] and flags[3]
