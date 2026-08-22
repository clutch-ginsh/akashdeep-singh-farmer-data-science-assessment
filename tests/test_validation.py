"""Tests for Layer 1 (physical bounds) and Layer 2 (OOD) guardrails."""

import pandas as pd

import wt_noise_lib as wtai
from wt_noise_lib import config
from wt_noise_lib.validation import OODDetector, validate_physical


def _row(**overrides):
    base = {
        "frequency_hz": 1000.0,
        "angle_of_attack_deg": 5.0,
        "chord_length_m": 0.1,
        "free_stream_velocity_ms": 55.5,
        "suction_displacement_m": 0.005,
        "sound_pressure_db": 125.0,
    }
    base.update(overrides)
    return base


def test_validate_physical_flags_impossible_aoa():
    df = pd.DataFrame([_row(), _row(angle_of_attack_deg=236758.0)])
    out = validate_physical(df)
    assert bool(out.loc[0, "is_physically_valid"]) is True
    assert bool(out.loc[1, "is_physically_valid"]) is False
    assert "angle_of_attack_deg" in out.loc[1, "violation"]


def test_validate_physical_flags_negative_frequency():
    df = pd.DataFrame([_row(frequency_hz=-10.0)])
    out = validate_physical(df)
    assert not out.loc[0, "is_physically_valid"]


def test_ood_detector_flags_out_of_domain_point():
    df = wtai.load_data()
    X = df[config.FEATURE_COLUMNS]
    det = OODDetector().fit(X)

    # An in-domain point (a real training row) should not be flagged.
    in_domain = X.iloc[[0]]
    assert not bool(det.predict(in_domain).iloc[0])

    # A physically valid but far-out-of-range AoA should be flagged OOD.
    ood = in_domain.copy()
    ood["angle_of_attack_deg"] = X["angle_of_attack_deg"].max() + 50.0
    assert bool(det.predict(ood).iloc[0])
