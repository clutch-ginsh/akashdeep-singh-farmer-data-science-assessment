# Wind-Tunnel Acoustic Anomaly Detection

**FARM Data Science Challenge — ML Engineer, Flight Physics**

Model the expected acoustic behaviour of an airfoil in a wind tunnel and detect
non-physical **instrument faults** in the sound-pressure-level (SPL) signal — the
kind of transient sensor glitches that currently force engineers to inspect raw
test logs by hand.

Dataset: [NASA / UCI Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)
(1,503 rows, 5 inputs → SPL in dB). The raw data is *clean*, so anomalies are
**synthetically injected** into the target to build a labelled test bed.

---

## Documentation

1. **This README** — how to set up and run the project.
2. **`run.py`** — the end-to-end analysis script
   (data analysis → modelling → anomaly injection → detection). It prints results,
   writes figures to `data/outputs/`, and builds an **interactive HTML report**
   (one per run) in the same folder.
3. **The `wt_noise_lib` package** (`src/`) — the tested, reusable core logic.

---

## Project layout

```
.
├── pyproject.toml                     # project metadata + dependencies (managed by uv)
├── README.md
├── run.py                             # end-to-end analysis script
├── data/input/airfoil_self_noise.dat  # raw dataset
├── data/outputs/run_<timestamp>/      # figures + interactive HTML report (one folder per run)
├── src/wt_noise_lib/                   # import name: wt_noise_lib (alias: wtai)
│   ├── config.py       # seeds, schema, physical bounds, injection settings
│   ├── data.py         # loading + grouped train/val/test split
│   ├── validation.py   # physical-bounds check + out-of-domain detector
│   ├── metrics.py      # regression + detection metrics
│   ├── models.py       # ML baselines + PyTorch MLPs
│   ├── anomaly.py      # synthetic injection + residual detector
│   └── utils.py        # reproducibility (set_seed)
└── tests/                             # pytest unit tests for the pure logic
```

## Setup & run

The project uses [**uv**](https://docs.astral.sh/uv/) — one tool that creates the
environment and installs the exact locked dependencies for you (no manual venv,
pip, or conda). [Install uv](https://docs.astral.sh/uv/getting-started/installation/),
then from the repo root:

```bash
# run the full analysis (uv builds the environment on first run)
uv run run.py

# run the unit tests
uv run pytest
```

Results print to the console; each run also writes its figures and a self-contained
**interactive HTML report** into its own folder `data/outputs/run_<timestamp>/`
(open `report.html` in any browser).

## Quickstart

```python
import wt_noise_lib as wtai

wtai.set_seed()
df = wtai.load_data()
train, val, test = wtai.grouped_split(df)
```

<!-- ### AI usage
AI coding assistance (Claude) was used for: scaffolding the package/notebook structure,
drafting docstrings, and design discussion (the guardrail framing, the
grouped-split leakage argument, and injection realism). **Manual review/refactoring:**
the XGBoost→HistGradientBoosting portability swap, verifying the grouped split is
leakage-free, confirming the threshold is calibrated on clean residuals only, and
rewriting the conclusions to match the *actual* measured results (DL narrowly winning)
rather than the assumed outcome. All modelling decisions were checked against held-out
metrics before being stated. -->
