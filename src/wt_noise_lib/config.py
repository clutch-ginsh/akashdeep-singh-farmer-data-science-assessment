"""Central configuration: seeds, schema, physical bounds, and anomaly settings.

Keeping these as named constants (rather than magic numbers scattered through
the analysis) makes every choice auditable and reproducible -- a requirement of
the challenge brief.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Data schema
# --------------------------------------------------------------------------- #
# Default location of the raw UCI/NASA dataset within the repo.
DATA_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "inputs" / "airfoil_self_noise.dat"

# Public source (documented for provenance).
DATA_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
)

# Column order as delivered in the raw tab-separated file (no header row).
FEATURE_COLUMNS: list[str] = [
    "frequency_hz",
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_ms",
    "suction_displacement_m",
]
TARGET_COLUMN: str = "sound_pressure_db"
COLUMNS: list[str] = FEATURE_COLUMNS + [TARGET_COLUMN]

# Columns that define a unique wind-tunnel *configuration* (everything except the
# swept frequency). Used for the grouped train/val/test split so that all
# frequency bands of one physical setup stay within a single fold -- prevents the
# near-duplicate-row leakage a naive random split would introduce.
CONFIG_COLUMNS: list[str] = [
    "angle_of_attack_deg",
    "chord_length_m",
    "free_stream_velocity_ms",
    "suction_displacement_m",
]

# --------------------------------------------------------------------------- #
# Physical validity bounds (hard, physics/spec-derived limits).
# Deliberately WIDER than the training data: these reject the physically
# impossible, not the merely unseen (that is the OOD detector's job).
# Each entry is (min, max) inclusive; None means unbounded on that side.
# --------------------------------------------------------------------------- #
PHYSICAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "frequency_hz": (0.0, None),            # audible acoustic band; must be positive
    "angle_of_attack_deg": (-90.0, 90.0),   # beyond +/-90 deg is not a meaningful AoA
    "chord_length_m": (0.0, None),          # a physical length; strictly positive
    "free_stream_velocity_ms": (0.0, None),  # non-negative flow speed
    "suction_displacement_m": (0.0, None),  # boundary-layer thickness; positive
    "sound_pressure_db": (0.0, 200.0),      # realistic SPL window for this rig
}

# --------------------------------------------------------------------------- #
# Split proportions
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.20  # fraction of the *train-remainder* held out for validation

# --------------------------------------------------------------------------- #
# Synthetic anomaly injection (Step 12).
# Severity -> additive spike magnitude in dB. Final values are reasoned from the
# residual/noise scale during the anomaly stage; these are the working defaults.
# --------------------------------------------------------------------------- #
INJECTION_FRACTION: float = 0.10  # fraction of the TEST set to corrupt
SEVERITY_DB: dict[str, tuple[float, float]] = {
    "low": (3.0, 6.0),
    "moderate": (6.0, 12.0),
    "severe": (12.0, 25.0),
}
