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

1. **This README** — how to install, run, and contribute to the project.
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

Requires Python ≥ 3.10. Use a **fresh, isolated environment** — `requirements.txt`
pins one mutually-compatible set of versions that runs identically on Linux,
macOS, and Windows (avoiding version/ABI mismatches such as the matplotlib
`_image` / Pillow `_imaging` import errors).

Pick **one** of the two options below.

### Option A — venv + pip

```bash
# 1. create & activate a clean environment
#    (use python3 on macOS/Linux; on Windows use: python -m venv .venv)
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# From here on, `python` and `pip` refer to the ones inside .venv.

# 2. install the exact pinned dependencies
pip install -r requirements.txt

# 3. install this project's package (no dependency re-resolution)
pip install -e . --no-deps

# 4. register the notebook kernel, then launch
python -m ipykernel install --user --name wt-noise --display-name "Python 3 (wt-noise)"
jupyter notebook notebooks/airfoil_anomaly_detection.ipynb
```

### Option B — conda


```bash
# 1. create & activate the environment (installs the pinned stack)
conda env create -f environment.yml
conda activate wt-noise

# 2. install this project's package + register the kernel, then launch
pip install -e . --no-deps
python -m ipykernel install --user --name wt-noise --display-name "Python 3 (wt-noise)"
jupyter notebook notebooks/airfoil_anomaly_detection.ipynb
```

**Important — run the notebook on the `wt-noise` kernel.** When the notebook opens,
select the **`Python 3 (wt-noise)`** kernel (Jupyter: *Kernel → Change Kernel*).

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
