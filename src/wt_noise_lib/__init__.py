"""wt_noise_lib -- Wind-tunnel acoustic anomaly detection.

Core logic for the Airbus FARM Data Science Challenge: modelling expected
airfoil self-noise (SPL) and detecting non-physical instrument faults.

Typical usage::

    import wt_noise_lib as wtai

    wtai.set_seed()
    df = wtai.load_data()
    train, val, test = wtai.grouped_split(df)
"""

from __future__ import annotations

from . import anomaly, config, data, metrics, models, utils, validation
from .anomaly import InjectionResult, ResidualAnomalyDetector, inject_spikes
from .data import config_group_ids, grouped_split, load_data
from .metrics import detection_metrics, regression_metrics
from .utils import set_seed
from .validation import OODDetector, validate_physical

__version__ = "0.1.0"

__all__ = [
    # submodules
    "anomaly",
    "config",
    "data",
    "metrics",
    "models",
    "utils",
    "validation",
    # convenience re-exports
    "load_data",
    "grouped_split",
    "config_group_ids",
    "validate_physical",
    "OODDetector",
    "inject_spikes",
    "InjectionResult",
    "ResidualAnomalyDetector",
    "regression_metrics",
    "detection_metrics",
    "set_seed",
    "__version__",
]
