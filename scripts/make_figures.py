"""Regenerate all paper figures from the result CSVs.

This script reads only the CSVs under ``results/`` and therefore needs no
raw ImageNet arrays or ProteinGym data.  Run it after any rerun of the
experiment scripts to keep the figures in step with the numbers.

    python scripts/make_figures.py

Outputs
-------
results/imagenet/ext1_main.png          coverage / MSE / ESS vs n
results/imagenet/interval_variants.png  variance-estimator comparison
results/proteingym/ext1_pg_main.png     coverage / MSE vs n
results/proteingym/ext1_pg_ranking.png  Spearman rho distribution
results/ablations/abl_combined.png      three ablation panels
results/diagnostics/dre_comparison.png  discriminative vs KMM/uLSIF/KLIEP
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

NOMINAL = 0.90
DPI = 160

# Consistent colours across every figure.
C_CLS = "#B3444A"   # classical
C_PPI = "#C8873C"   # unweighted PPI++
C_ORA = "#6B7280"   # oracle
C_ADA = "#2F6F8F"   # Adaptive AutoEval


def _nominal(ax):
    ax.axhline(NOMINAL, ls="--", lw=1.0, c="0.35", zorder=1)
    ax.text(
        0.99, NOMINAL, " nominal 0.90",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7, color="0.35",
    )


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


def imagenet_main():
    df = pd.read_csv(os.path.join(RES, "imagenet", "ext1_results.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    ax.plot(df.n, df.cov_cls, "o-", c=C_CLS, label="Classical")
    ax.plot(df.n, df.cov_ppi, "s-", c=C_PPI, label="PPI++ (unweighted)")
    ax.plot(df.n, df.cov_oppi, "^-", c=C_ORA, label="Oracle PPI++")
    ax.plot(df.n, df.cov_wppi, "D-", c=C_ADA, lw=2, label="Adaptive AutoEval")
    _nominal(ax)
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("coverage of 90% CI")
    ax.set_title("Coverage")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[1]
    ax.plot(df.n, df.mse_cls, "o-", c=C_CLS, label="Classical")
    ax.plot(df.n, df.mse_wppi, "D-", c=C_ADA, lw=2, label="Adaptive AutoEval")
    ax.set_yscale("log")
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Mean squared error")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    ax.plot(df.n, df.ess_wppi, "D-", c=C_ADA, lw=2)
    ax.axhline(1.0, ls="--", lw=1.0, c="0.35")
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("ESS ratio")
    ax.set_title("Effective sample size")

    fig.suptitle(
        "ImageNet -- annotator-independent weight features, exponential shift "
        r"($\beta$=1.0)",
        y=1.04, fontsize=10,
    )
    _finish(fig, os.path.join(RES, "imagenet", "ext1_main.png"))


def imagenet_interval_variants():
    df = pd.read_csv(os.path.join(RES, "imagenet", "interval_variants.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = {
        "two_term": "two-term (default)",
        "bootstrap": "weighted bootstrap",
        "crossfit": "5-fold cross-fitting",
        "t_quantile": "t-quantile",
        "crossfit_t": "cross-fitting + t-quantile",
    }
    for col, lab in labels.items():
        ax.plot(df.n, df[col], "o-", ms=4, lw=1.4, label=lab)
    _nominal(ax)
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("coverage of 90% CI")
    ax.set_title("ImageNet: variance-estimator variants")
    ax.set_ylim(0.78, 0.93)
    ax.legend(fontsize=7.5, ncol=2)
    _finish(fig, os.path.join(RES, "imagenet", "interval_variants.png"))


def proteingym_main():
    df = pd.read_csv(os.path.join(RES, "proteingym", "ext1_pg_results.csv"))
    dc = pd.read_csv(os.path.join(RES, "proteingym", "interval_variants.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    ax.plot(df.n, df.cov_cls, "o-", c=C_CLS, label="Classical")
    ax.plot(df.n, df.cov_ppi, "s-", c=C_PPI, label="PPI++ (unweighted)")
    ax.plot(df.n, df.cov_wppi, "D-", c=C_ADA, lw=2, label="Adaptive AutoEval")
    _nominal(ax)
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("coverage of 90% CI")
    ax.set_title("Coverage -- annotator-coupled features")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7.5, loc="center right")

    ax = axes[1]
    ax.plot(dc.n, dc.two_term, "D-", c=C_ADA, lw=2, label="Adaptive AutoEval")
    _nominal(ax)
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("coverage of 90% CI")
    ax.set_title("Coverage -- annotator-independent features")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7.5)

    ax = axes[2]
    ax.plot(df.n, df.mse_cls, "o-", c=C_CLS, label="Classical")
    ax.plot(df.n, df.mse_wppi, "D-", c=C_ADA, lw=2, label="Adaptive AutoEval")
    ax.set_yscale("log")
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Mean squared error")
    ax.legend(fontsize=7.5)

    fig.suptitle(
        "ProteinGym SPG1 -- weight-feature choice dominates the result. "
        "Annotator-independent features (centre) make weight estimation harder.",
        y=1.04, fontsize=9.5,
    )
    _finish(fig, os.path.join(RES, "proteingym", "ext1_pg_main.png"))


def proteingym_ranking():
    df = pd.read_csv(os.path.join(RES, "proteingym", "ext1_pg_spearman.csv"))
    cols = [("spearman_cls", "Classical", C_CLS),
            ("spearman_ppi", "PPI++", C_PPI),
            ("spearman_wppi", "Adaptive AutoEval", C_ADA)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    parts = ax.violinplot([df[c] for c, _, _ in cols], showmeans=True)
    for body, (_, _, colour) in zip(parts["bodies"], cols):
        body.set_facecolor(colour)
        body.set_alpha(0.55)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([lab for _, lab, _ in cols])
    ax.set_ylabel(r"Spearman $\rho$ vs ground-truth ranking")
    ax.set_title("ProteinGym model ranking (100 trials, n=600)")
    for i, (c, _, _) in enumerate(cols, start=1):
        ax.annotate(f"mean {df[c].mean():.3f}", (i, df[c].mean()),
                    textcoords="offset points", xytext=(12, 0),
                    fontsize=7.5, va="center")
    _finish(fig, os.path.join(RES, "proteingym", "ext1_pg_ranking.png"))


def ablations():
    beta = pd.read_csv(os.path.join(RES, "ablations", "abl_shift_severity.csv"))
    clf = pd.read_csv(os.path.join(RES, "ablations", "abl_classifier_capacity.csv"))
    clip = pd.read_csv(os.path.join(RES, "ablations", "abl_clipping_threshold.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    for b, g in beta.groupby("beta"):
        ax.plot(g.n, g.cov_wppi, "o-", ms=4, label=rf"$\beta$={b}")
    _nominal(ax)
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("coverage of 90% CI")
    ax.set_title("Shift severity")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for name, g in clf.groupby("classifier"):
        ax.plot(g.n, g.cov_wppi, "o-", ms=4, label=name)
    _nominal(ax)
    ax.set_xlabel("labeled sample size $n$")
    ax.set_title("Weight-classifier capacity")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    for eps, g in clip.groupby("eps"):
        ax.plot(g.n, g.cov_wppi, "o-", ms=4, label=rf"$\epsilon$={eps}")
    _nominal(ax)
    ax.set_xlabel("labeled sample size $n$")
    ax.set_title("Clipping threshold")
    ax.legend(fontsize=7.5, ncol=2)

    fig.suptitle(
        "ImageNet ablations -- exponential shift, annotator-independent features",
        y=1.04, fontsize=10,
    )
    _finish(fig, os.path.join(RES, "ablations", "abl_combined.png"))


def dre_comparison():
    df = pd.read_csv(os.path.join(RES, "diagnostics", "dre_comparison_imagenet.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    methods = [("Discriminative", C_ADA), ("KMM", "#4C8C5A"),
               ("uLSIF", C_PPI), ("KLIEP", C_CLS)]

    ax = axes[0]
    for m, colour in methods:
        ax.plot(df.n, df[f"cov_{m}"], "o-", ms=4, c=colour, label=m)
    _nominal(ax)
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("coverage of 90% CI")
    ax.set_title("Coverage by density-ratio estimator")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for m, colour in methods:
        ax.plot(df.n, df[f"winrate_{m}"], "o-", ms=4, c=colour, label=m)
    ax.set_xlabel("labeled sample size $n$")
    ax.set_ylabel("per-trial win rate")
    ax.set_title("Fraction of trials each estimator wins")
    ax.legend(fontsize=7.5)

    fig.suptitle(
        "Alternative density-ratio estimators (matched low-dimensional features, "
        "identical clipping)", y=1.04, fontsize=10,
    )
    _finish(fig, os.path.join(RES, "diagnostics", "dre_comparison.png"))


def main():
    print("Regenerating figures from result CSVs ...")
    imagenet_main()
    imagenet_interval_variants()
    proteingym_main()
    proteingym_ranking()
    ablations()
    dre_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
