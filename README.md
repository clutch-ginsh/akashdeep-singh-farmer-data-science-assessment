# Wind-Tunnel Acoustic Anomaly Detection

**Airbus FARM Data Science Challenge — ML Engineer, Flight Physics**
Author: Akashdeep Singh

Model the expected acoustic behaviour of an airfoil in a wind tunnel and detect
non-physical **instrument faults** in the sound-pressure-level (SPL) signal — the
kind of transient sensor glitches that currently force engineers to inspect raw
test logs by hand.

Dataset: [NASA / UCI Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)
(1,503 rows, 5 inputs → SPL in dB). The raw data is *clean*, so anomalies are
**synthetically injected** into the target to build a labelled test bed.

---

## Documentation strategy

Three artefacts, each for a different reader:

1. **This README** — how to install, run, and reason about the project (the map).
2. **The notebook** `notebooks/airfoil_anomaly_detection.ipynb` — the narrative:
   EDA → modelling → anomaly injection → detection → conclusions. It *imports*
   the library rather than defining logic inline, so it stays readable.
3. **The `wt_noise_lib` package** (`src/`) — the tested, reusable core logic.
   Heavy lifting lives here so it can be unit-tested (a notebook cannot be).

A **decision log** at the bottom of this file records why each modelling and
architectural choice was made, and where AI assistance was used.

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

---

## Approach in one picture — defence in depth

The word "anomaly" hides three distinct failure modes, each handled separately:

| Layer | Question | Failure mode | Mechanism |
|-------|----------|--------------|-----------|
| 1 | Is the input physically possible? | invalid input (e.g. AoA = 236758°) | hard physical bounds (`validate_physical`) |
| 2 | Is the input inside the training domain? | covariate shift / extrapolation | OOD detector on features (`OODDetector`) |
| 3 | Is the **SPL reading** anomalous? | acoustic instrument fault | regression **residual** vs threshold (`ResidualAnomalyDetector`) |

The brief strictly requires **Layer 3**; Layers 1–2 are production guardrails
that make the real-time-stream story credible.

---

## Decision log

Records the *why*, not just the *what*.

**Architecture**
- **Library + thin notebook.** Core logic lives in the importable, unit-tested
  `wt_noise_lib` package; the notebook is a readable narrative that imports it. A
  notebook cannot be `pytest`-ed — a module can.
- **HistGradientBoosting instead of XGBoost.** Same LightGBM-style gradient-boosted
  trees, but ships with scikit-learn and needs no Homebrew `libomp`. This keeps the
  deliverable installable from `pip install .` alone on any evaluator's machine.

**Data & evaluation**
- **Split — grouped by configuration.** Rows sharing a physical setup differ only in
  the swept frequency; a naive random split would scatter these near-duplicates across
  folds and inflate scores. We group by `(AoA, chord, velocity, displacement)` so
  evaluation is always on unseen configurations (honest generalisation).
- **Scaling — features (and NN target) standardised, fit on train only.** Trees are
  scale-invariant and use raw features; neural nets need conditioned inputs. The NN
  target is standardised and inverse-transformed before any residual is computed.
- **Metrics fixed before modelling.** Regression: RMSE/MAE/R² (dB-interpretable).
  Detection: PR-AUC as the headline (anomalies are rare) plus ROC-AUC and P/R/F1.

**Modelling & anomalies**
- **Anomaly injection — target-only, additive dB spikes, seeded.** Matches the brief
  ("low/moderate/severe noise spikes"); inputs stay clean, so the residual method is
  the natural detector. Magnitudes are sized in units of the model's clean residual σ.
- **Detection threshold — calibrated on clean (val) residuals only.** Never on the
  injected test set, which would leak the anomalies we aim to detect.
- **Dedicated detector uses the joint `(features, SPL)` space**, not the residual, so
  the comparison against Layer 3 is fair rather than circular.

**Evidence-based outcomes** (reproducible, seed = 42)
- **Deep learning is justified here — but only just.** The deeper PyTorch MLP gave the
  best test RMSE (≈ 3.1 dB, R² ≈ 0.80) vs RandomForest/HistGB (≈ 3.7–3.9 dB) and linear
  (≈ 5.0 dB). Likely because SPL is a smooth continuous surface suited to an MLP; the
  ~0.8 dB margin over the far simpler trees means production may still prefer a tree.
- **Simplest detector wins.** The residual method (PR-AUC ≈ 0.67) beats an
  IsolationForest on the joint space (≈ 0.22): faults are target-only, so residuals are
  near-optimal. Severe faults are caught 100%, moderate ~64%, low (1–2σ) mostly missed.

### AI usage
AI coding assistance (Claude) was used for: scaffolding the package/notebook structure,
drafting docstrings, and design discussion (the three-layer guardrail framing, the
grouped-split leakage argument, and injection realism). **Manual review/refactoring:**
the XGBoost→HistGradientBoosting portability swap, verifying the grouped split is
leakage-free, confirming the threshold is calibrated on clean residuals only, and
rewriting the conclusions to match the *actual* measured results (DL narrowly winning)
rather than the assumed outcome. All modelling decisions were checked against held-out
metrics before being stated.
