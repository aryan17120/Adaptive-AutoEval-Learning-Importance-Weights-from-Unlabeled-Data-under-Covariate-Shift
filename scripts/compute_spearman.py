"""
Compute exact Spearman rank correlations for ProteinGym ranking reliability.
Runs 100 trials at n=600 and reports:
  - Mean Spearman correlation between estimated and ground-truth model rankings
  - Standard deviation across trials
  - One representative single-trial ranking for the figure

Save as: scripts/compute_spearman.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG — same as full experiment
# --------------------------------------------------
DATA_PATH   = "data/proteingym/SPG1_STRSG_Olson_2014_zero_shot.csv"
N_CALIB     = 10000
N_UNLABELED = 10000
BETA_SHIFT  = 1.0
N_SHOW      = 600          # the n=600 used in the ranking plot
N_TRIALS    = 100          # 100 trials gives stable mean Spearman
SEED        = 42
ALPHA       = 0.10

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

np.random.seed(SEED)

# --------------------------------------------------
# Load & Setup (identical to full experiment)
# --------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
needed = ["DMS_score", ANNOTATOR] + list(TARGET_MODELS.values())
df = df.dropna(subset=needed).reset_index(drop=True)
N_total = len(df)
print(f"Variants: {N_total}")

# Split calibration / pool
perm      = np.random.RandomState(SEED).permutation(N_total)
calib_idx = perm[:N_CALIB]
pool_idx  = perm[N_CALIB:]

# Fit calibrations
Y_calib = df["DMS_score"].values[calib_idx]
calibrations = {}
for name, col in TARGET_MODELS.items():
    x   = df[col].values[calib_idx].reshape(-1, 1)
    reg = LinearRegression().fit(x, Y_calib)
    calibrations[name] = {"col": col,
                           "alpha": reg.coef_[0],
                           "beta":  reg.intercept_}

x_ann   = df[ANNOTATOR].values[calib_idx].reshape(-1, 1)
reg_ann = LinearRegression().fit(x_ann, Y_calib)
ann_cal = {"alpha": reg_ann.coef_[0], "beta": reg_ann.intercept_}

# Calibrated predictions & squared errors on pool
Y_pool    = df["DMS_score"].values[pool_idx]
Y_hat     = {name: calibrations[name]["alpha"] *
                   df[calibrations[name]["col"]].values[pool_idx]
                   + calibrations[name]["beta"]
             for name in TARGET_MODELS}
Y_hat_ann = ann_cal["alpha"] * df[ANNOTATOR].values[pool_idx] + ann_cal["beta"]

se_true = {name: (Y_hat[name] - Y_pool) ** 2 for name in TARGET_MODELS}
se_syn  = {name: (Y_hat[name] - Y_hat_ann) ** 2 for name in TARGET_MODELS}

# Ground truth MSEs (target = uniform pool)
mu_gt = np.array([se_true[name].mean() for name in TARGET_MODELS])
model_names = list(TARGET_MODELS.keys())

print("\nGround-truth MSEs (target = uniform):")
for i, name in enumerate(model_names):
    print(f"  {name:8s}: {mu_gt[i]:.4f}")

# Fitness shift weights
Y_norm  = (Y_pool - Y_pool.mean()) / Y_pool.std()
log_p   = BETA_SHIFT * Y_norm
log_p  -= log_p.max()
p_shift = np.exp(log_p); p_shift /= p_shift.sum()

# Feature matrix for weight learning
feats_pool = np.hstack([
    np.column_stack([df[c["col"]].values[pool_idx]
                     for c in calibrations.values()]),
    df[ANNOTATOR].values[pool_idx].reshape(-1, 1),
    (ann_cal["alpha"] * df[ANNOTATOR].values[pool_idx]
     + ann_cal["beta"]).reshape(-1, 1),
])

# --------------------------------------------------
# Estimators (identical to full experiment)
# --------------------------------------------------
def ppi_unweighted(phi_lab, syn_lab, syn_unl):
    n = len(phi_lab); N = len(syn_unl)
    cov_num  = np.cov(phi_lab, syn_lab, ddof=0)[0, 1]
    var_full = (n / N) * syn_unl.var() + syn_lab.var()
    lam      = np.clip(cov_num / (var_full + 1e-12), 0.0, 1.0)
    return lam * syn_unl.mean() + (phi_lab - lam * syn_lab).mean()


def ppi_weighted(phi_lab, syn_lab, syn_unl, weights):
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
    # Weight the full residual (φ − λÊ)
    return lam * syn_unl.mean() + (w * (phi_lab - lam * syn_lab)).mean()


def learn_weights(feat_lab, feat_unl):
    n = len(feat_lab); N = len(feat_unl)
    if N > 5 * n:
        feat_unl = feat_unl[np.random.choice(N, size=5*n, replace=False)]
    X  = np.vstack([feat_lab, feat_unl])
    y  = np.array([0]*n + [1]*len(feat_unl))
    sc = StandardScaler().fit(X)
    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=500, C=1.0, random_state=0),
        cv=3, method="sigmoid",
    )
    clf.fit(sc.transform(X), y)
    p_unl = np.clip(clf.predict_proba(sc.transform(feat_lab))[:, 1], 0.01, 0.99)
    w     = p_unl / (1.0 - p_unl)
    return np.clip(w, 0.01, 100.0) / np.clip(w, 0.01, 100.0).mean()

# --------------------------------------------------
# Run trials and collect per-model estimates
# --------------------------------------------------
print(f"\nRunning {N_TRIALS} trials at n={N_SHOW}...")
print("(Estimating Spearman rank correlations)")

spearman_cls  = []
spearman_ppi  = []
spearman_wppi = []

# Also store one representative trial for the ranking plot
rep_mu_c = rep_mu_p = rep_mu_w = None
rep_trial = 0   # use trial 0 as the representative

for trial in range(N_TRIALS):
    rng = np.random.RandomState(trial * 1000 + N_SHOW)

    idx_lab   = rng.choice(len(pool_idx), size=N_SHOW,
                           replace=False, p=p_shift)
    remaining = np.setdiff1d(np.arange(len(pool_idx)), idx_lab)
    idx_unl   = rng.choice(remaining, size=N_UNLABELED, replace=False)

    try:
        w = learn_weights(feats_pool[idx_lab], feats_pool[idx_unl])
    except Exception:
        w = np.ones(N_SHOW)

    mu_c_trial = []
    mu_p_trial = []
    mu_w_trial = []

    for name in model_names:
        phi_lab = se_true[name][idx_lab]
        syn_lab = se_syn [name][idx_lab]
        syn_unl = se_syn [name][idx_unl]

        mu_c_trial.append(phi_lab.mean())
        mu_p_trial.append(ppi_unweighted(phi_lab, syn_lab, syn_unl))
        mu_w_trial.append(ppi_weighted  (phi_lab, syn_lab, syn_unl, w))

    mu_c_trial = np.array(mu_c_trial)
    mu_p_trial = np.array(mu_p_trial)
    mu_w_trial = np.array(mu_w_trial)

    # Compute Spearman correlation with ground truth ranking
    rho_c,  _ = spearmanr(mu_c_trial,  mu_gt)
    rho_p,  _ = spearmanr(mu_p_trial,  mu_gt)
    rho_w,  _ = spearmanr(mu_w_trial,  mu_gt)

    spearman_cls.append(rho_c)
    spearman_ppi.append(rho_p)
    spearman_wppi.append(rho_w)

    if trial == rep_trial:
        rep_mu_c = mu_c_trial.copy()
        rep_mu_p = mu_p_trial.copy()
        rep_mu_w = mu_w_trial.copy()

    if (trial + 1) % 20 == 0:
        print(f"  Trial {trial+1}/{N_TRIALS} | "
              f"ρ_cls={np.mean(spearman_cls):.3f}  "
              f"ρ_ppi={np.mean(spearman_ppi):.3f}  "
              f"ρ_wppi={np.mean(spearman_wppi):.3f}")

# --------------------------------------------------
# Results
# --------------------------------------------------
spearman_cls  = np.array(spearman_cls)
spearman_ppi  = np.array(spearman_ppi)
spearman_wppi = np.array(spearman_wppi)

print("\n" + "=" * 60)
print(f"SPEARMAN RANK CORRELATIONS AT n={N_SHOW}")
print(f"(Mean ± Std over {N_TRIALS} trials)")
print("=" * 60)
print(f"  Classical:         {spearman_cls.mean():.3f} ± {spearman_cls.std():.3f}")
print(f"  PPI++ (unweighted):{spearman_ppi.mean():.3f} ± {spearman_ppi.std():.3f}")
print(f"  Adaptive AutoEval: {spearman_wppi.mean():.3f} ± {spearman_wppi.std():.3f}")

print("\nGround-truth model ranking (best to worst, lower MSE = better):")
gt_order = np.argsort(mu_gt)
for rank, idx in enumerate(gt_order):
    print(f"  #{rank+1}: {model_names[idx]:8s}  MSE={mu_gt[idx]:.4f}")

print("\nRepresentative trial (trial 0) per-model estimates:")
print(f"  {'Model':10s} | {'GT':>7s} | {'Classical':>10s} | "
      f"{'PPI++':>10s} | {'Adaptive':>10s}")
print("  " + "-" * 58)
for i, name in enumerate(model_names):
    print(f"  {name:10s} | {mu_gt[i]:>7.4f} | "
          f"{rep_mu_c[i]:>10.4f} | "
          f"{rep_mu_p[i]:>10.4f} | "
          f"{rep_mu_w[i]:>10.4f}")

print("\nRepresentative trial Spearman correlations:")
rho_c_rep, _ = spearmanr(rep_mu_c, mu_gt)
rho_p_rep, _ = spearmanr(rep_mu_p, mu_gt)
rho_w_rep, _ = spearmanr(rep_mu_w, mu_gt)
print(f"  Classical:         {rho_c_rep:.3f}")
print(f"  PPI++ (unweighted):{rho_p_rep:.3f}")
print(f"  Adaptive AutoEval: {rho_w_rep:.3f}")

# --------------------------------------------------
# Save results
# --------------------------------------------------
import os
out_dir = "results/extension1_proteingym"
os.makedirs(out_dir, exist_ok=True)

results_df = pd.DataFrame({
    "trial":         np.arange(N_TRIALS),
    "spearman_cls":  spearman_cls,
    "spearman_ppi":  spearman_ppi,
    "spearman_wppi": spearman_wppi,
})
results_df.to_csv(os.path.join(out_dir,
                  "ext1_pg_spearman.csv"), index=False)
print(f"\nSaved: {out_dir}/ext1_pg_spearman.csv")

# --------------------------------------------------
# Print paper-ready sentence
# --------------------------------------------------
print("\n" + "=" * 60)
print("PAPER-READY SENTENCE:")
print("=" * 60)
rho_c_m  = spearman_cls.mean()
rho_p_m  = spearman_ppi.mean()
rho_w_m  = spearman_wppi.mean()
rho_c_s  = spearman_cls.std()
rho_p_s  = spearman_ppi.std()
rho_w_s  = spearman_wppi.std()

print(f"""
Quantitatively, Adaptive AutoEval achieves a Spearman rank correlation of
{rho_w_m:.2f} (±{rho_w_s:.2f}) with the true model ranking at n={N_SHOW},
compared to {rho_c_m:.2f} (±{rho_c_s:.2f}) for the Classical estimator
and {rho_p_m:.2f} (±{rho_p_s:.2f}) for PPI++.
""")
print("Done.")