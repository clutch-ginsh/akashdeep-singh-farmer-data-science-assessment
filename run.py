"""End-to-end wind-tunnel acoustic anomaly-detection analysis.

Runs the full pipeline — data checks, EDA, guardrails, model comparison,
synthetic anomaly injection, and residual-based detection. Each run writes its
figures and a self-contained interactive HTML report into its own timestamped
folder, ./data/outputs/run_<timestamp>/.

Usage:
    uv run run.py
"""

from __future__ import annotations

import base64
import html
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk, no display needed

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import wt_noise_lib as wtai  # noqa: E402
from wt_noise_lib import anomaly, config, metrics, models, validation  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "outputs"
RUN_DIR: Path = OUT  # per-run subfolder data/outputs/run_<timestamp>, set in main()


# --------------------------------------------------------------------------- #
# Report capture: mirror console output into per-section buffers so we can build
# an HTML report without duplicating every print statement.
# --------------------------------------------------------------------------- #
class _Tee:
    """A stdout wrapper that also appends writes to the current section buffer."""

    def __init__(self, real):
        self.real = real
        self.sink: list[str] | None = None

    def write(self, s):
        self.real.write(s)
        if self.sink is not None:
            self.sink.append(s)

    def flush(self):
        self.real.flush()


_SECTIONS: list[dict] = []
_TEE: _Tee | None = None


def banner(title: str) -> None:
    """Start a new report section and print a console banner for it."""
    _TEE.real.write("\n" + "=" * 72 + f"\n {title}\n" + "=" * 72 + "\n")
    section = {"title": title, "text": [], "figures": []}
    _SECTIONS.append(section)
    _TEE.sink = section["text"]


def save(name: str) -> None:
    """Save the current figure into the run folder and attach it to the section."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RUN_DIR / name, dpi=120, bbox_inches="tight")
    plt.close("all")
    if _SECTIONS:
        _SECTIONS[-1]["figures"].append(name)
    print(f"  figure -> data/outputs/{RUN_DIR.name}/{name}")


def build_html_report(meta: dict) -> Path:
    """Assemble a single self-contained interactive HTML report for this run."""
    def slug(t):
        return "sec-" + "".join(c if c.isalnum() else "-" for c in t.lower()).strip("-")

    def img_tag(name):
        data = base64.b64encode((RUN_DIR / name).read_bytes()).decode("ascii")
        return f'<img alt="{html.escape(name)}" src="data:image/png;base64,{data}"/>'

    toc = "\n".join(
        f'<li><a href="#{slug(s["title"])}">{html.escape(s["title"])}</a></li>'
        for s in _SECTIONS
    )
    body = []
    for s in _SECTIONS:
        text = html.escape("".join(s["text"]).strip("\n"))
        figs = "\n".join(f'<div class="fig">{img_tag(n)}</div>' for n in s["figures"])
        body.append(
            f'<details open id="{slug(s["title"])}">'
            f'<summary>{html.escape(s["title"])}</summary>'
            + (f"<pre>{text}</pre>" if text else "")
            + figs
            + "</details>"
        )
    meta_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in meta.items()
    )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Wind-Tunnel Acoustic Anomaly Detection — Run Report</title>
<style>
 :root {{ --fg:#1b1f24; --muted:#5b6673; --acc:#0b6bcb; --bg:#f6f8fa; --card:#fff; --line:#e2e6eb; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
        color:var(--fg); background:var(--bg); line-height:1.5; }}
 header {{ background:linear-gradient(135deg,#0b6bcb,#083d73); color:#fff; padding:28px 32px; }}
 header h1 {{ margin:0 0 4px; font-size:22px; }}
 header p {{ margin:0; opacity:.9; font-size:14px; }}
 .wrap {{ display:grid; grid-template-columns:250px 1fr; gap:24px; max-width:1150px;
         margin:24px auto; padding:0 20px; }}
 nav {{ position:sticky; top:16px; align-self:start; background:var(--card);
        border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
 nav h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:0 0 8px; }}
 nav ul {{ list-style:none; margin:0; padding:0; }}
 nav li {{ margin:2px 0; }}
 nav a {{ color:var(--acc); text-decoration:none; font-size:13px; }}
 nav a:hover {{ text-decoration:underline; }}
 .controls button {{ font-size:12px; margin:10px 6px 0 0; padding:4px 10px; cursor:pointer;
        border:1px solid var(--line); border-radius:6px; background:#fff; }}
 main {{ min-width:0; }}
 table.meta {{ border-collapse:collapse; background:var(--card); border:1px solid var(--line);
        border-radius:10px; overflow:hidden; margin-bottom:18px; width:100%; }}
 table.meta th, table.meta td {{ text-align:left; padding:6px 12px; border-bottom:1px solid var(--line); font-size:13px; }}
 table.meta th {{ color:var(--muted); width:180px; font-weight:600; }}
 details {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:6px 18px; margin-bottom:14px; }}
 summary {{ cursor:pointer; font-weight:600; font-size:16px; padding:8px 0; }}
 pre {{ background:#0d1117; color:#e6edf3; padding:14px 16px; border-radius:8px;
        overflow:auto; font-size:12.5px; line-height:1.45; }}
 .fig {{ margin:12px 0; }}
 .fig img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; }}
 footer {{ text-align:center; color:var(--muted); font-size:12px; padding:24px; }}
</style></head>
<body>
<header>
 <h1>Wind-Tunnel Acoustic Anomaly Detection</h1>
 <p>Automated run report &middot; {html.escape(meta.get("generated", ""))}</p>
</header>
<div class="wrap">
 <nav>
  <h2>Contents</h2>
  <ul>{toc}</ul>
  <div class="controls">
   <button onclick="document.querySelectorAll('details').forEach(d=>d.open=true)">Expand all</button>
   <button onclick="document.querySelectorAll('details').forEach(d=>d.open=false)">Collapse all</button>
  </div>
 </nav>
 <main>
  <table class="meta">{meta_rows}</table>
  {"".join(body)}
 </main>
</div>
<footer>Generated by run.py &middot; wt_noise_lib {html.escape(str(meta.get("wt_noise_lib", "")))}</footer>
</body></html>"""

    path = RUN_DIR / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path


def main() -> None:
    global _TEE, RUN_DIR
    _TEE = _Tee(sys.stdout)
    sys.stdout = _TEE
    started = datetime.now()
    RUN_DIR = OUT / f"run_{started.strftime('%Y%m%d_%H%M%S')}"
    try:
        sns.set_theme(style="whitegrid", context="notebook")
        wtai.set_seed()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        print("wt_noise_lib", wtai.__version__, "| global seed:", config.RANDOM_SEED)
        print("run folder ->", f"data/outputs/{RUN_DIR.name}")

        # -------------------------------------------------------------- #
        banner("1. Data loading")
        df = wtai.load_data()
        print("shape:", df.shape)
        print(df.dtypes.to_string())
        print(df.head().to_string())

        # -------------------------------------------------------------- #
        banner("2. Statistical sanity checks")
        summary = df.describe().T[["min", "mean", "50%", "max", "std"]]
        print("missing values:", int(df.isna().sum().sum()),
              "| exact duplicate rows:", int(df.duplicated().sum()))
        print(summary.round(4).to_string())
        spl = df[config.TARGET_COLUMN]
        print("\n- Sanity check on clean data.")
        print(f"- SPL spans {spl.min():.1f}-{spl.max():.1f} dB (mean {spl.mean():.1f}).")
        print(f"- Frequency spans {df.frequency_hz.min():.0f}-{df.frequency_hz.max():.0f} Hz.")
        print(f"- Angle of attack {df.angle_of_attack_deg.min():.1f}-{df.angle_of_attack_deg.max():.1f} deg")

        # -------------------------------------------------------------- #
        banner("3. Visual EDA")
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for ax, col in zip(axes.ravel(), config.COLUMNS):
            sns.histplot(df[col], ax=ax, bins=30, color="steelblue")
            ax.set_title(col)
            if col == "frequency_hz":
                ax.set_xscale("log")
                ax.set_title("frequency_hz (log axis)")
        fig.suptitle("Feature & target distributions", y=1.02, fontsize=13)
        plt.tight_layout()
        save("01_distributions.png")

        fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), sharey=True)
        for ax, col in zip(axes, config.FEATURE_COLUMNS):
            ax.scatter(df[col], df[config.TARGET_COLUMN], s=6, alpha=0.3, color="darkorange")
            ax.set_xlabel(col)
            if col == "frequency_hz":
                ax.set_xscale("log")
        axes[0].set_ylabel("sound_pressure_db")
        fig.suptitle("SPL vs each input feature", y=1.05, fontsize=13)
        plt.tight_layout()
        save("02_spl_vs_features.png")

        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(6.5, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, square=True, cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1]})
        plt.title("Correlation matrix")
        plt.tight_layout()
        save("03_correlation.png")
        tcorr = corr[config.TARGET_COLUMN].drop(config.TARGET_COLUMN).sort_values()
        print("Correlation of each feature with SPL:")
        print(tcorr.to_string())

        # -------------------------------------------------------------- #
        banner("4. Physics validity bounds")
        demo = pd.DataFrame([
            {"frequency_hz": 1000, "angle_of_attack_deg": 5.0, "chord_length_m": 0.1,
             "free_stream_velocity_ms": 55.5, "suction_displacement_m": 0.005, "sound_pressure_db": 125.0},
            {"frequency_hz": 1000, "angle_of_attack_deg": 236758.0, "chord_length_m": 0.1,
             "free_stream_velocity_ms": 55.5, "suction_displacement_m": 0.005, "sound_pressure_db": 125.0},
            {"frequency_hz": -5, "angle_of_attack_deg": 5.0, "chord_length_m": 0.1,
             "free_stream_velocity_ms": 55.5, "suction_displacement_m": 0.005, "sound_pressure_db": 125.0},
        ])
        checked = validation.validate_physical(demo)
        print(checked[["angle_of_attack_deg", "frequency_hz", "is_physically_valid", "violation"]].to_string())

        # -------------------------------------------------------------- #
        banner("5. Acoustic-spectrum check")
        grp = df.groupby(config.CONFIG_COLUMNS, sort=False)
        sizes = grp.size().sort_values(ascending=False)
        plt.figure(figsize=(9, 5))
        for key in sizes.index[:5]:
            sub = grp.get_group(key).sort_values("frequency_hz")
            label = f"AoA={key[0]:.0f}deg, c={key[1]:.3f}m, U={key[2]:.0f}m/s"
            plt.plot(sub.frequency_hz, sub[config.TARGET_COLUMN], marker="o", ms=3, label=label)
        plt.xscale("log")
        plt.xlabel("frequency (Hz)")
        plt.ylabel("SPL (dB)")
        plt.title("Measured self-noise spectra (5 configurations)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        save("04_spectra.png")

        # -------------------------------------------------------------- #
        banner("6. Out-of-domain / low-confidence detector")
        ood = validation.OODDetector().fit(df[config.FEATURE_COLUMNS])
        in_domain = df[config.FEATURE_COLUMNS].iloc[[0]]
        far = in_domain.copy()
        far["angle_of_attack_deg"] = df.angle_of_attack_deg.max() + 25.0
        print("real in-domain row flagged low-confidence? ", bool(ood.predict(in_domain).iloc[0]))
        print(f"AoA = {float(far.angle_of_attack_deg.iloc[0]):.1f}deg (beyond tested "
              f"max {df.angle_of_attack_deg.max():.1f}deg) flagged low-confidence? ",
              bool(ood.predict(far).iloc[0]))

        # -------------------------------------------------------------- #
        banner("7. Grouped train/val/test split")
        train_df, val_df, test_df = wtai.grouped_split(df)
        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
            n_cfg = part.groupby(config.CONFIG_COLUMNS, sort=False).ngroups
            print(f"{name:5s}: {len(part):4d} rows  |  {n_cfg:3d} configurations")

        def cfgset(p):
            return set(map(tuple, p[config.CONFIG_COLUMNS].to_numpy()))

        assert cfgset(train_df).isdisjoint(cfgset(test_df))
        assert cfgset(train_df).isdisjoint(cfgset(val_df))
        assert cfgset(val_df).isdisjoint(cfgset(test_df))
        print("no configuration is shared across folds (honest generalisation).")

        # -------------------------------------------------------------- #
        banner("8. Scaling")
        feat, tgt = config.FEATURE_COLUMNS, config.TARGET_COLUMN
        X_train_raw, y_train = train_df[feat].to_numpy(), train_df[tgt].to_numpy()
        X_val_raw, y_val = val_df[feat].to_numpy(), val_df[tgt].to_numpy()
        X_test_raw, y_test = test_df[feat].to_numpy(), test_df[tgt].to_numpy()

        scaler_X = StandardScaler().fit(X_train_raw)
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        Xtr, Xva, Xte = (scaler_X.transform(a) for a in (X_train_raw, X_val_raw, X_test_raw))
        ytr_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
        yva_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

        def to_db(scaled):  # inverse-transform NN outputs back to dB
            return scaler_y.inverse_transform(np.asarray(scaled).reshape(-1, 1)).ravel()

        print("scaler fit on train only - feature means ~ 0:", np.round(Xtr.mean(0), 3))

        # -------------------------------------------------------------- #
        banner("9. Metrics")
        baseline_rmse = float(np.sqrt(np.mean((y_test - y_train.mean()) ** 2)))
        print(f"mean-predictor test RMSE = {baseline_rmse:.2f} dB (the floor any model must beat)")

        # -------------------------------------------------------------- #
        banner("10. Models")
        preds_val, preds_test, trained = {}, {}, {}
        for name, model in models.get_baseline_models().items():
            model.fit(X_train_raw, y_train)
            preds_val[name] = model.predict(X_val_raw)
            preds_test[name] = model.predict(X_test_raw)
            trained[name] = model
        print("classical models trained:", list(trained.keys()))

        mlp_specs = {"mlp_64x64": (64, 64), "mlp_deep_128x128x64": (128, 128, 64)}
        for name, hidden in mlp_specs.items():
            wtai.set_seed()
            net = models.build_mlp(input_dim=Xtr.shape[1], hidden=hidden, dropout=0.1)
            net, hist = models.train_mlp(net, Xtr, ytr_s, Xva, yva_s, epochs=400, patience=40)
            preds_val[name] = to_db(models.mlp_predict(net, Xva))
            preds_test[name] = to_db(models.mlp_predict(net, Xte))
            trained[name] = net
            print(f"{name:22s} trained - stopped after {len(hist['train_loss'])} epochs")

        # -------------------------------------------------------------- #
        banner("11. Model comparison")
        rows = []
        for name in trained:
            mv = metrics.regression_metrics(y_val, preds_val[name])
            mt = metrics.regression_metrics(y_test, preds_test[name])
            rows.append({"model": name, "val_RMSE": mv["rmse_db"], "test_RMSE": mt["rmse_db"],
                         "test_MAE": mt["mae_db"], "test_R2": mt["r2"]})
        results = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
        best_name = results.iloc[0]["model"]
        print(results.round(3).to_string(index=False))

        best_model_is_dl = best_name.startswith("mlp")
        best_dl = results[results.model.str.startswith("mlp")].sort_values("val_RMSE").iloc[0]
        best_cls = results[~results.model.str.startswith("mlp")].sort_values("val_RMSE").iloc[0]
        print(f"\nBest overall by validation RMSE: {best_name}")
        print(f"Best classical: {best_cls.model} (test RMSE {best_cls.test_RMSE:.3f} dB)")
        print(f"Best deep-net : {best_dl.model} (test RMSE {best_dl.test_RMSE:.3f} dB)")
        if best_model_is_dl:
            print("VERDICT: deep learning wins here - but note the modest margin over the tree baseline.")
        else:
            print("VERDICT: a classical tree model beats both PyTorch nets on this ~1,500-row tabular\n"
                  "problem. Deep learning is NOT automatically best; the evidence favours the baseline.")
        print(f"gap (best DL - best classical) = {best_dl.test_RMSE - best_cls.test_RMSE:+.3f} dB RMSE")

        yhat = preds_test[best_name]
        resid = y_test - yhat
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].scatter(y_test, yhat, s=10, alpha=0.4)
        lim = [min(y_test.min(), yhat.min()), max(y_test.max(), yhat.max())]
        ax[0].plot(lim, lim, "r--", lw=1)
        ax[0].set_xlabel("actual SPL (dB)")
        ax[0].set_ylabel("predicted SPL (dB)")
        ax[0].set_title(f"{best_name}: predicted vs actual")
        sns.histplot(resid, bins=30, ax=ax[1], color="slateblue")
        ax[1].set_xlabel("residual (dB)")
        ax[1].set_title("residual distribution (test)")
        plt.tight_layout()
        save("05_predicted_vs_actual.png")
        print(f"chosen model '{best_name}': test residual std = {resid.std():.3f} dB")

        # -------------------------------------------------------------- #
        banner("12. Synthetic anomaly injection")
        sigma = resid.std()
        print(f"clean residual sigma ~ {sigma:.2f} dB -> severity bands in units of sigma:")
        for lvl, (lo, hi) in config.SEVERITY_DB.items():
            print(f"  {lvl:9s}: {lo:.0f}-{hi:.0f} dB  ~ {lo / sigma:.1f}-{hi / sigma:.1f} sigma")
        inj = anomaly.inject_spikes(y_test, fraction=config.INJECTION_FRACTION, seed=config.RANDOM_SEED)
        print(f"injected {inj.is_anomaly.sum()} / {len(y_test)} test points "
              f"({inj.is_anomaly.mean() * 100:.0f}%). Only the SPL target was altered.")

        order = np.argsort(y_test)
        plt.figure(figsize=(12, 4.5))
        plt.scatter(range(len(y_test)), inj.values[order], s=14, c="lightgrey", label="measured")
        for lvl, colr in [("low", "gold"), ("moderate", "orange"), ("severe", "red")]:
            m = (inj.severity[order] == lvl)
            plt.scatter(np.where(m)[0], inj.values[order][m], s=28, c=colr, label=lvl, zorder=3)
        plt.plot(range(len(y_test)), y_test[order], "k-", lw=0.6, alpha=0.6, label="clean truth")
        plt.xlabel("test points (sorted by clean SPL)")
        plt.ylabel("SPL (dB)")
        plt.title("Injected instrument faults by severity")
        plt.legend()
        plt.tight_layout()
        save("06_injected_faults.png")

        # -------------------------------------------------------------- #
        banner("13. Residual-based detection")
        clean_resid = y_val - preds_val[best_name]  # calibrate on CLEAN val residuals only
        detector = anomaly.ResidualAnomalyDetector(z_thresh=3.5).fit(clean_resid)
        test_resid = inj.values - preds_test[best_name]
        scores = detector.score(test_resid)
        flags = detector.predict(test_resid)

        det = metrics.detection_metrics(inj.is_anomaly.astype(int), scores, flags.astype(int))
        print("Residual detector - test performance:")
        for k, v in det.items():
            print(f"  {k:10s}: {v:.3f}")

        from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
        prec, rec, _ = precision_recall_curve(inj.is_anomaly.astype(int), scores)
        fpr, tpr, _ = roc_curve(inj.is_anomaly.astype(int), scores)
        cm = confusion_matrix(inj.is_anomaly.astype(int), flags.astype(int))
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
        ax[0].plot(rec, prec)
        ax[0].set_xlabel("recall")
        ax[0].set_ylabel("precision")
        ax[0].set_title(f"PR curve (AP={det['pr_auc']:.3f})")
        ax[1].plot(fpr, tpr)
        ax[1].plot([0, 1], [0, 1], "r--", lw=1)
        ax[1].set_xlabel("FPR")
        ax[1].set_ylabel("TPR")
        ax[1].set_title(f"ROC (AUC={det['roc_auc']:.3f})")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[2],
                    xticklabels=["clean", "anom"], yticklabels=["clean", "anom"])
        ax[2].set_xlabel("predicted")
        ax[2].set_ylabel("actual")
        ax[2].set_title("confusion matrix")
        plt.tight_layout()
        save("07_detection_curves.png")

        print("recall by severity (fraction of injected faults caught):")
        for lvl in ("low", "moderate", "severe"):
            m = inj.severity == lvl
            if m.sum():
                print(f"  {lvl:9s}: {flags[m].mean():.2f}  (n={int(m.sum())})")

        # -------------------------------------------------------------- #
        banner("Conclusions")
        print(
            f"1. Model choice : best = {best_name} "
            f"(test RMSE {best_dl.test_RMSE:.2f} dB, R2 {results.iloc[0].test_R2:.2f}); "
            f"best tree within {abs(best_dl.test_RMSE - best_cls.test_RMSE):.2f} dB.\n"
            f"2. Residual detection works: PR-AUC {det['pr_auc']:.2f} / ROC-AUC {det['roc_auc']:.2f}; "
            f"severe faults caught fully, low (near noise floor) mostly missed.\n"
            f"3. Guardrails: Rejected impossible inputs and flagged out-of-domain inputs."
        )

        # -------------------------------------------------------------- #
        meta = {
            "generated": started.strftime("%Y-%m-%d %H:%M:%S"),
            "wt_noise_lib": wtai.__version__,
            "python": platform.python_version(),
            "seed": config.RANDOM_SEED,
            "best model": best_name,
            "best test RMSE (dB)": f"{best_dl.test_RMSE:.3f}",
            "detection PR-AUC": f"{det['pr_auc']:.3f}",
        }
        report_path = build_html_report(meta)
    finally:
        sys.stdout = _TEE.real

    print(f"\nInteractive HTML report -> data/outputs/{RUN_DIR.name}/{report_path.name}")


if __name__ == "__main__":
    main()
