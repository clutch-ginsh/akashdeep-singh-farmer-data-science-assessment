# Wind-Tunnel Acoustic Anomaly Detection

**FARM Data Science Challenge — ML Engineer, Flight Physics**
Author: Akashdeep Singh

Model the expected acoustic behaviour of an airfoil in a wind tunnel and detect
non-physical **instrument faults** in the sound-pressure-level (SPL) signal — the
kind of transient sensor glitches that currently force engineers to inspect raw
test logs by hand.

Dataset: [NASA / UCI Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)
(1,503 rows, 5 inputs → SPL in dB). The raw data is *clean*, so anomalies are
**synthetically injected** into the target to build a labelled test bed.

---

## Documentation

1. **This README** — how to install, run, and reason about the project (the map).
2. **The notebook** `notebooks/airfoil_anomaly_detection.ipynb`:
   Data analysis → modelling → anomaly injection → detection → conclusions.
3. **The `wt_noise_lib` package** (`src/`) — the tested, reusable core logic.

---

## Project layout

```
.
├── pyproject.toml                     # pip-installable (distribution: akashdeep-singh-farmer-data-science-assessment)
├── README.md
├── data/input/airfoil_self_noise.dat  # raw dataset
├── notebooks/
│   └── airfoil_anomaly_detection.ipynb
├── src/wt_noise_lib/                   # import name: wt_noise_lib (alias: wtai)
│   ├── config.py       # seeds, schema, physical bounds, injection settings
│   ├── data.py         # loading + grouped train/val/test split
│   ├── validation.py   # Layer 1 physical bounds + Layer 2 OOD detector
│   ├── metrics.py      # regression + detection metrics
│   ├── models.py       # ML baselines + PyTorch MLPs
│   ├── anomaly.py      # synthetic injection + residual detector
│   └── utils.py        # reproducibility (set_seed)
└── tests/                             # pytest unit tests for the pure logic
```

## Setup

Requires Python ≥ 3.10.

```bash
# create and activate an environment (example)
python -m venv .venv && source .venv/bin/activate

# users: install the library
pip install .

# developers: editable install + dev tools (pytest, jupyter)
pip install -e ".[dev]"

# fallback: pinned exact versions from a verified environment
pip install -r requirements.txt
```

Run the tests:

```bash
pytest
```

Launch the notebook:

```bash
jupyter notebook notebooks/airfoil_anomaly_detection.ipynb
```

## Quickstart

```python
import wt_noise_lib as wtai

wtai.set_seed()
df = wtai.load_data()
train, val, test = wtai.grouped_split(df)
```

<!-- ### AI usage
AI coding assistance (Claude) was used for: scaffolding the package/notebook structure,
drafting docstrings, and design discussion (the three-layer guardrail framing, the
grouped-split leakage argument, and injection realism). **Manual review/refactoring:**
the XGBoost→HistGradientBoosting portability swap, verifying the grouped split is
leakage-free, confirming the threshold is calibrated on clean residuals only, and
rewriting the conclusions to match the *actual* measured results (DL narrowly winning)
rather than the assumed outcome. All modelling decisions were checked against held-out
metrics before being stated. -->
