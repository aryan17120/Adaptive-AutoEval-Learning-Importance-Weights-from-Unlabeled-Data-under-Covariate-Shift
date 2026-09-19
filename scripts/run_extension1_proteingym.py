"""
Extension 1: Adaptive AutoEval on ProteinGym — FULL Monte Carlo.

Dataset: SPG1_STRSG_Olson_2014 (536k protein variants, DMS fitness assay)
Task:    Estimate MSE of 7 foundation models on the target distribution
Shift:   Fitness-biased labeling (high-fitness variants overrepresented)

Three estimators:
  (1) Classical  — labeled mean, biased under shift
  (2) PPI++      — unweighted, partially corrects via synthetic annotator
  (3) Adaptive   — our method with learned importance weights

Outputs:
  results/extension1_proteingym/ext1_pg_results.csv
  results/extension1_proteingym/ext1_pg_main.png
  results/extension1_proteingym/ext1_pg_ranking.png
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_PATH = "data/proteingym/SPG1_STRSG_Olson_2014_zero_shot.csv"
OUT_DIR   = "results/extension1_proteingym"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_MODELS = {
    "CARP":      "CARP_640M",
    "ESM-1b":    "ESM1b",
    "ESM-1v":    "ESM1v_ensemble",
    "ESM-2":     "ESM2_650M",
    "ProGen2":   "Progen2_large",
    "RITA":      "RITA_l",
    "UniRep":    "Unirep",
}
ANNOTATOR = "VESPA"

BETA_SHIFT  = 1.0
N_CALIB     = 10000
N_UNLABELED = 10000
N_LIST      = [200, 400, 600, 800, 1000, 1500]
N_TRIALS    = 250
ALPHA       = 0.10          # 90% CIs
SEED        = 42
np.random.seed(SEED)

# --------------------------------------------------
# Load data
# --------------------------------------------------
print("=" * 74)
print("Extension 1: Adaptive AutoEval on ProteinGym (full MC)")
print("=" * 74)

print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
needed_cols = ["DMS_score", ANNOTATOR] + list(TARGET_MODELS.values())
df = df.dropna(subset=needed_cols).reset_index(drop=True)
print(f"Variants (non-NaN): {len(df)}")
N_total = len(df)

# --------------------------------------------------
# Split: calibration / pool
# --------------------------------------------------
perm = np.random.RandomState(SEED).permutation(N_total)
calib_idx = perm[:N_CALIB]
pool_idx  = perm[N_CALIB:]

# Fit calibrations
Y_calib = df["DMS_score"].values[calib_idx]
calibrations = {}
for name, col in TARGET_MODELS.items():
    x = df[col].values[calib_idx].reshape(-1, 1)
    reg = LinearRegression().fit(x, Y_calib)
    calibrations[name] = {"col": col, "alpha": reg.coef_[0],
                          "beta": reg.intercept_}
x_ann = df[ANNOTATOR].values[calib_idx].reshape(-1, 1)
reg_ann = LinearRegression().fit(x_ann, Y_calib)
ann_cal = {"alpha": reg_ann.coef_[0], "beta": reg_ann.intercept_}

# Calibrated predictions on pool
Y_pool   = df["DMS_score"].values[pool_idx]
Y_hat    = {name: calibrations[name]["alpha"] * df[calibrations[name]["col"]].values[pool_idx]
                 + calibrations[name]["beta"]
           for name in TARGET_MODELS}
Y_hat_ann = ann_cal["alpha"] * df[ANNOTATOR].values[pool_idx] + ann_cal["beta"]

se_true = {name: (Y_hat[name] - Y_pool) ** 2 for name in TARGET_MODELS}
se_syn  = {name: (Y_hat[name] - Y_hat_ann) ** 2 for name in TARGET_MODELS}

# Ground truth (per model)
mu_gt = np.array([se_true[name].mean() for name in TARGET_MODELS])
print("\nGround-truth MSEs (target = uniform over pool):")
for i, name in enumerate(TARGET_MODELS):
    print(f"  {name:8s} : {mu_gt[i]:.4f}")

# Fitness shift
Y_norm = (Y_pool - Y_pool.mean()) / Y_pool.std()
log_p  = BETA_SHIFT * Y_norm
log_p -= log_p.max()
p_shift = np.exp(log_p); p_shift /= p_shift.sum()

# Weight-estimation features must be independent of the variable that drives
# the selection bias. Here Y_norm drives both the shift and the annotator, so
# including the VESPA annotator score would let the discriminator separate
# labeled from unlabeled trivially, inflating apparent performance.
#
# Preferred: ESM-2 640-d embeddings (App. D.2), independent of the fitness
# direction for the binary labeled/unlabeled task.
# Fallback: raw (uncalibrated) model scores, less directly tied to fitness.
ESM2_FEAT_PATH = f"results/features_proteingym/esm2_embeddings.npy"
if os.path.exists(ESM2_FEAT_PATH):
    _esm2_all = np.load(ESM2_FEAT_PATH)   # shape (len(df), 640)
    feats_pool = _esm2_all[pool_idx]
    FEATURE_SOURCE_PG = "esm2_640d"
else:
    # Fallback: raw (uncalibrated) model predictions only, no annotator column.
    # Excludes VESPA to remove the direct annotator-feature coupling.
    feats_pool = np.column_stack([
        df[c["col"]].values[pool_idx] for c in calibrations.values()
    ])
    FEATURE_SOURCE_PG = "raw_model_scores_no_annotator"
    print(f"  ESM-2 embeddings not found. Using fallback features: {FEATURE_SOURCE_PG}")
    print(f"  Run scripts/extract_features_proteingym.py to generate ESM-2 embeddings.")

# --------------------------------------------------
# Estimators
# --------------------------------------------------
def ppi_unweighted(phi_lab, syn_lab, syn_unl):
    n = len(phi_lab); N = len(syn_unl)
    cov_num  = np.cov(phi_lab, syn_lab, ddof=0)[0, 1]
    var_full = (n / N) * syn_unl.var() + syn_lab.var()
    lam      = np.clip(cov_num / (var_full + 1e-12), 0.0, 1.0)
    mu_hat   = lam * syn_unl.mean() + (phi_lab - lam * syn_lab).mean()
    resid    = phi_lab - lam * syn_lab
    var_hat  = resid.var() / n + lam**2 * syn_unl.var() * (n / N) / n
    return mu_hat, var_hat


def ppi_weighted(phi_lab, syn_lab, syn_unl, weights):
    # Weight the full residual (φ−λÊ); λ* from the Appendix B weighted
    # moments; var_hat from the same residual as mu_hat.
    n = len(phi_lab); N = len(syn_unl)
    w = weights / weights.mean()
    # Appendix B weighted moments
    phi_bar_w = (w * phi_lab).mean()
    syn_bar_w = (w * syn_lab).mean()
    cov_w    = (w * (phi_lab - phi_bar_w) * (syn_lab - syn_bar_w)).mean()
    var_w    = (w * (syn_lab - syn_bar_w) ** 2).mean()
    var_unl  = syn_unl.var()
    denom    = var_w + (n / N) * var_unl
    lam      = np.clip(cov_w / (denom + 1e-12), 0.0, 1.0)
    # Shared residual
    resid    = w * (phi_lab - lam * syn_lab)
    mu_hat   = lam * syn_unl.mean() + resid.mean()
    var_hat  = resid.var() / n + lam**2 * var_unl * (n / N) / n
    return mu_hat, var_hat


def learn_weights(feat_lab, feat_unl):
    n = len(feat_lab); N = len(feat_unl)
    if N > 5 * n:
        rng = np.random.RandomState(0)
        feat_unl = feat_unl[rng.choice(N, size=5 * n, replace=False)]
    X  = np.vstack([feat_lab, feat_unl])
    y  = np.array([0]*n + [1]*len(feat_unl))
    sc = StandardScaler().fit(X)
    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=500, C=1.0, random_state=0),
        cv=3, method="sigmoid",
    )
    clf.fit(sc.transform(X), y)
    p_unl = np.clip(clf.predict_proba(sc.transform(feat_lab))[:, 1], 0.01, 0.99)
    w = p_unl / (1.0 - p_unl)
    return np.clip(w, 0.01, 100.0) / np.clip(w, 0.01, 100.0).mean()

# --------------------------------------------------
# Monte Carlo
# --------------------------------------------------
print(f"\nRunning {N_TRIALS} trials × {len(N_LIST)} sample sizes...")
print("(Expect ~10-15 min)")
print("-" * 74)

results = []

for n in N_LIST:
    cov_cls, cov_ppi, cov_wppi = [], [], []
    wid_cls, wid_ppi, wid_wppi = [], [], []
    mse_cls, mse_ppi, mse_wppi = [], [], []
    ess_ppi, ess_wppi          = [], []

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial * 1000 + n)

        # Sample
        idx_lab = rng.choice(len(pool_idx), size=n, replace=False, p=p_shift)
        remaining = np.setdiff1d(np.arange(len(pool_idx)), idx_lab)
        idx_unl = rng.choice(remaining, size=N_UNLABELED, replace=False)

        # Weights
        try:
            w = learn_weights(feats_pool[idx_lab], feats_pool[idx_unl])
        except Exception:
            w = np.ones(n)

        # Per-model estimates
        mu_c_all, mu_p_all, mu_w_all = [], [], []
        var_c_all, var_p_all, var_w_all = [], [], []

        for name in TARGET_MODELS:
            phi_lab = se_true[name][idx_lab]
            syn_lab = se_syn [name][idx_lab]
            syn_unl = se_syn [name][idx_unl]

            mu_c  = phi_lab.mean()
            var_c = phi_lab.var() / n

            mu_p, var_p = ppi_unweighted(phi_lab, syn_lab, syn_unl)
            mu_w, var_w = ppi_weighted  (phi_lab, syn_lab, syn_unl, w)

            mu_c_all.append(mu_c);   var_c_all.append(var_c)
            mu_p_all.append(mu_p);   var_p_all.append(var_p)
            mu_w_all.append(mu_w);   var_w_all.append(var_w)

        mu_c_all = np.array(mu_c_all); mu_p_all = np.array(mu_p_all); mu_w_all = np.array(mu_w_all)
        var_c_all = np.array(var_c_all); var_p_all = np.array(var_p_all); var_w_all = np.array(var_w_all)

        z = norm.ppf(1 - ALPHA / 2)
        def cov_fn(mu, var):
            return float(np.mean((mu_gt >= mu - z*np.sqrt(np.abs(var))) &
                                 (mu_gt <= mu + z*np.sqrt(np.abs(var)))))
        def wid_fn(var):
            return float(np.mean(2 * z * np.sqrt(np.abs(var))))

        cov_cls.append(cov_fn(mu_c_all, var_c_all))
        cov_ppi.append(cov_fn(mu_p_all, var_p_all))
        cov_wppi.append(cov_fn(mu_w_all, var_w_all))
        wid_cls.append(wid_fn(var_c_all))
        wid_ppi.append(wid_fn(var_p_all))
        wid_wppi.append(wid_fn(var_w_all))
        mse_cls.append(float(np.mean((mu_c_all - mu_gt)**2)))
        mse_ppi.append(float(np.mean((mu_p_all - mu_gt)**2)))
        mse_wppi.append(float(np.mean((mu_w_all - mu_gt)**2)))
        ess_ppi.append(float(np.mean(np.abs(var_c_all)/(np.abs(var_p_all)+1e-12))))
        ess_wppi.append(float(np.mean(np.abs(var_c_all)/(np.abs(var_w_all)+1e-12))))

    sm = lambda l: float(np.mean(l))
    row = dict(n=n,
               cov_cls=sm(cov_cls), cov_ppi=sm(cov_ppi), cov_wppi=sm(cov_wppi),
               wid_cls=sm(wid_cls), wid_ppi=sm(wid_ppi), wid_wppi=sm(wid_wppi),
               mse_cls=sm(mse_cls), mse_ppi=sm(mse_ppi), mse_wppi=sm(mse_wppi),
               ess_ppi=sm(ess_ppi), ess_wppi=sm(ess_wppi))
    results.append(row)

    print(f"n={n:4d} | cov cls={row['cov_cls']:.3f}  "
          f"ppi={row['cov_ppi']:.3f}  wppi={row['cov_wppi']:.3f} | "
          f"mse_cls={row['mse_cls']:.4f}  mse_wppi={row['mse_wppi']:.4f} | "
          f"ESS wppi={row['ess_wppi']:.2f}")

# --------------------------------------------------
# Save
# --------------------------------------------------
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUT_DIR, "ext1_pg_results.csv"), index=False)
print(f"\nSaved: {OUT_DIR}/ext1_pg_results.csv")
print(df_res.to_string(index=False))

# --------------------------------------------------
# Figures
# --------------------------------------------------
import matplotlib.pyplot as plt

ns = df_res["n"].values
CC, CP, CW = "#4DAF4A", "#E41A1C", "#377EB8"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(ns, df_res["cov_cls"],  "o-",  color=CC, label="Classical")
ax.plot(ns, df_res["cov_ppi"],  "s--", color=CP, label="PPI++ (unweighted)")
ax.plot(ns, df_res["cov_wppi"], "o-",  color=CW, label="Adaptive AutoEval")
ax.axhline(1 - ALPHA, color="k", ls=":", lw=1.2, label="Target 90%")
ax.set_xlabel("# labeled mutations"); ax.set_ylabel("Coverage of 90% CIs")
ax.set_title("(a) Coverage under fitness shift")
ax.set_ylim(0.0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(ns, df_res["mse_cls"],  "o-",  color=CC, label="Classical")
ax.plot(ns, df_res["mse_ppi"],  "s--", color=CP, label="PPI++ (unweighted)")
ax.plot(ns, df_res["mse_wppi"], "o-",  color=CW, label="Adaptive AutoEval")
ax.set_xlabel("# labeled mutations"); ax.set_ylabel("MSE of estimate")
ax.set_title("(b) MSE of per-model MSE estimates")
ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(ns, df_res["ess_ppi"],  "s--", color=CP, label="PPI++ (unweighted)")
ax.plot(ns, df_res["ess_wppi"], "o-",  color=CW, label="Adaptive AutoEval")
ax.axhline(1.0, color=CC, ls="--", lw=1.0, label="Classical (=1)")
ax.set_xlabel("# labeled mutations"); ax.set_ylabel("ESS")
ax.set_title("(c) ESS under fitness shift")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(
    f"Extension 1: Adaptive AutoEval on ProteinGym (SPG1 DMS) — "
    f"fitness-shift β={BETA_SHIFT}, annotator = VESPA\n"
    f"Metric: Mean squared error of calibrated foundation-model predictions",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ext1_pg_main.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {OUT_DIR}/ext1_pg_main.png")

# --------------------------------------------------
# Ranking plot at n=600
# --------------------------------------------------
print("Generating ranking reliability plot...")
n_show = 600
rng = np.random.RandomState(0)
idx_lab = rng.choice(len(pool_idx), size=n_show, replace=False, p=p_shift)
remaining = np.setdiff1d(np.arange(len(pool_idx)), idx_lab)
idx_unl = rng.choice(remaining, size=N_UNLABELED, replace=False)
w = learn_weights(feats_pool[idx_lab], feats_pool[idx_unl])

mu_c_plot, mu_p_plot, mu_w_plot = [], [], []
for name in TARGET_MODELS:
    phi_lab = se_true[name][idx_lab]
    syn_lab = se_syn [name][idx_lab]
    syn_unl = se_syn [name][idx_unl]
    mu_c_plot.append(phi_lab.mean())
    mu_p_plot.append(ppi_unweighted(phi_lab, syn_lab, syn_unl)[0])
    mu_w_plot.append(ppi_weighted  (phi_lab, syn_lab, syn_unl, w)[0])

fig2, ax2 = plt.subplots(figsize=(11, 5))
x = np.arange(len(TARGET_MODELS))
width = 0.2
ax2.bar(x - 1.5*width, mu_gt,      width, label="Ground truth", color="k", alpha=0.7)
ax2.bar(x - 0.5*width, mu_c_plot,  width, label="Classical",   color=CC)
ax2.bar(x + 0.5*width, mu_p_plot,  width, label="PPI++",       color=CP)
ax2.bar(x + 1.5*width, mu_w_plot,  width, label="Adaptive",    color=CW)
ax2.set_xticks(x)
ax2.set_xticklabels(list(TARGET_MODELS.keys()), fontsize=10)
ax2.set_ylabel("Mean squared error")
ax2.set_title(f"Foundation model MSEs on SPG1 (n={n_show}, β={BETA_SHIFT})")
ax2.legend(); ax2.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ext1_pg_ranking.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved: {OUT_DIR}/ext1_pg_ranking.png")

# --------------------------------------------------
# Summary
# --------------------------------------------------
print("\n" + "=" * 74)
print("SUMMARY")
print("=" * 74)
print(f"{'n':>5} | {'cov_cls':>7} | {'cov_ppi':>7} | {'cov_wppi':>8} | "
      f"{'MSE_cls':>8} | {'MSE_wppi':>8} | {'ESS':>5}")
print("-" * 74)
for _, r in df_res.iterrows():
    print(f"{int(r.n):>5} | {r.cov_cls:>7.3f} | {r.cov_ppi:>7.3f} | "
          f"{r.cov_wppi:>8.3f} | {r.mse_cls:>8.4f} | {r.mse_wppi:>8.4f} | "
          f"{r.ess_wppi:>5.2f}")
print("\nDone.")