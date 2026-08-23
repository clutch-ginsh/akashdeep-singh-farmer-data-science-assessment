"""Regression models: classical ML baselines and PyTorch neural nets.

The classical baselines establish an honest floor and a strong tabular
reference; the PyTorch models give deep learning a fair, well-built shot. The
analysis compares them on held-out metrics to decide -- with evidence -- whether
deep learning is actually the right tool for ~1,500 tabular rows.
"""

from __future__ import annotations

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Classical ML baselines (train on RAW features -- trees are scale-invariant)
# --------------------------------------------------------------------------- #
def get_baseline_models(seed: int = config.RANDOM_SEED) -> dict:
    """Return a dict of untrained scikit-learn regressors.

    Ladder: LinearRegression (trivial floor) -> RandomForest -> HistGradientBoosting.

    ``HistGradientBoostingRegressor`` is a LightGBM-style gradient-boosted-tree
    model that ships with scikit-learn and needs no system libraries -- chosen
    over XGBoost so the project runs under ``uv run`` with no Homebrew ``libomp``
    dependency (portability for the evaluator).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    return {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "hist_gbdt": HistGradientBoostingRegressor(
            max_iter=400,
            max_depth=4,
            learning_rate=0.05,
            random_state=seed,
        ),
    }


# --------------------------------------------------------------------------- #
# PyTorch neural nets (train on SCALED features + scaled target)
# --------------------------------------------------------------------------- #
def build_mlp(input_dim: int, hidden: tuple[int, ...] = (64, 64), dropout: float = 0.1):
    """Build a configurable MLP regressor (returns an ``nn.Module``).

    Two instances cover the brief's "best DL model + an alternative": a plain
    MLP and a deeper/wider variant, selected by the ``hidden`` argument.
    """
    import torch.nn as nn

    layers: list = []
    prev = input_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_mlp(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 30,
    seed: int = config.RANDOM_SEED,
):
    """Train an MLP with Adam + MSE and early stopping on validation RMSE.

    Inputs are expected pre-scaled. Returns ``(model, history)`` where history
    is a dict of per-epoch train/val losses. Predictions are in the (scaled)
    target space -- inverse-transform before computing dB residuals.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    def _t(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32)).reshape(len(a), -1)

    train_ds = TensorDataset(_t(X_train), _t(y_train))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    Xv, yv = _t(X_val).to(device), _t(y_val).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val, best_state, waited = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        history["train_loss"].append(running / len(train_ds))
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_state, waited = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def mlp_predict(model, X: np.ndarray) -> np.ndarray:
    """Predict with a trained MLP; returns a 1-D array in the model's output space.

    Inputs must be scaled the same way as training; the caller inverse-transforms
    back to dB when the target was standardised.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(X, dtype=np.float32)).reshape(len(X), -1).to(device)
        return model(t).cpu().numpy().ravel()
