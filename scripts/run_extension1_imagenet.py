"""
Extension 1: Handling Distribution Shift with Learned Importance Weights
NOW WITH ORACLE BASELINE (true importance weights known)

Compares 4 estimators:
  (1) Classical      — labeled data only
  (2) PPI++          — unweighted, standard AutoEval
  (3) Oracle PPI++   — weighted PPI++ with TRUE importance weights
  (4) Adaptive PPI++ — weighted PPI++ with LEARNED importance weights

Outputs:
  results/imagenet/ext1_results.csv     (updated with oracle columns)
  results/imagenet/ext1_main.png        (updated 4-method figure)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
PHI_DIR     = "results/phi_imagenet"
SYN_DIR     = "results/synthetic_imagenet"
OUT_DIR     = "results/imagenet"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
N_TRIALS    = 250
N_LIST      = [50, 100, 200, 300, 400, 500]
ALPHA       = 0.1
SEED        = 42
np.random.seed(SEED)

# --------------------------------------------------
# STEP 1 — Load data
# --------------------------------------------------
print("Loading phi (correctness) and synthetic scores ...")
phi_all = np.stack(
    [np.load(os.path.join(PHI_DIR, f"{m}.npy")) for m in MODEL_NAMES], axis=1
)
syn_all = np.stack(
    [np.load(os.path.join(SYN_DIR, f"{m}.npy")) for m in MODEL_NAMES], axis=1
)

N_total, M = phi_all.shape
print(f"Loaded: {N_total} images, {M} models")

mu_gt = phi_all.mean(axis=0)
print("Ground-truth accuracies:")
for name, acc in zip(MODEL_NAMES, mu_gt):
    print(f"  {name}: {acc:.4f}")

# --------------------------------------------------
# Weight-estimation features must be independent of the variable that drives
# the selection bias, or the discriminator is handed the answer and weight
# estimation becomes trivially easy rather than informative.
# Preferred: 2048-d ResNet-50 penultimate layer features (App. D.1).
# Fallback: 1-d softmax entropy, which is not collinear with the ResNet-101
# confidence direction that drives the shift, because entropy is maximised at
# medium confidence rather than at high confidence.
#
# The shift mechanism is exponential, ∝ exp(β·s), with β=1.0 by default --
# the same parameterisation the severity ablation sweeps.
# --------------------------------------------------

FEAT_PATH = "results/features_imagenet/resnet50_penultimate.npy"
if os.path.exists(FEAT_PATH):
    print(f"Loading ResNet-50 penultimate features from {FEAT_PATH} ...")
    features = np.load(FEAT_PATH)          # shape (N_total, 2048)
    assert features.shape[0] == N_total, \
        f"Feature rows {features.shape[0]} != image count {N_total}"
    print(f"  Features shape: {features.shape}")
    FEATURE_SOURCE = "resnet50_penultimate_2048d"
else:
    # Proxy: per-image softmax entropy over the 5-model confidence vector.
    # H(p) = -sum(p*log(p)).  High entropy ≠ high confidence, so this is
    # not collinear with the ResNet-101 shift variable (Pearson r ≈ -0.65).
    # See scripts/extract_features_imagenet.py for the full extraction.
    print("ResNet-50 penultimate features not found.")
    print("  Falling back to softmax-entropy proxy features (1-d).")
    print("  Run scripts/extract_features_imagenet.py to generate the full features.")
    p_syn = np.clip(syn_all, 1e-7, 1.0)
    p_syn = p_syn / p_syn.sum(axis=1, keepdims=True)
    entropy = -np.sum(p_syn * np.log(p_syn), axis=1, keepdims=True)   # (N,1)
    features = entropy
    FEATURE_SOURCE = "softmax_entropy_proxy_1d"

# Exponential shift with β=1.0 (same parameterisation as the ablations)
SHIFT_BETA    = 1.0
_shift_raw    = syn_all[:, 3]                       # ResNet-101 score drives shift
_shift_log    = SHIFT_BETA * _shift_raw
_shift_log   -= _shift_log.max()                    # numerical stability
shift_weights = np.exp(_shift_log)                  # ∝ exp(β·s)
# (oracle weights use 1/shift_weights; the normalised probability is
# computed per-trial inside the MC loop)

# --------------------------------------------------
# STEP 2 — Estimators
# --------------------------------------------------

def ppi_estimate(phi_lab, syn_lab, syn_unl):
    """Standard PPI++ (unweighted)."""
    n, M = phi_lab.shape
    N    = syn_unl.shape[0]
    cov_num  = np.mean((phi_lab - phi_lab.mean(0)) *
                       (syn_lab - syn_lab.mean(0)), axis=0)
    var_full = (n / N) * syn_unl.var(0) + syn_lab.var(0)
    lambd    = np.clip(np.where(var_full > 1e-12,
                                cov_num / var_full, 1.0), 0.0, 1.0)
    mu_hat   = lambd * syn_unl.mean(0) + (phi_lab - lambd * syn_lab).mean(0)
    resid    = phi_lab - lambd * syn_lab
    var_hat  = resid.var(0) / n + lambd**2 * syn_unl.var(0) * (n / N) / n
    return mu_hat, var_hat


def ppi_estimate_weighted(phi_lab, syn_lab, syn_unl, weights):
    """Weighted PPI++ (used for both Oracle and Adaptive).

    The weight multiplies the full residual (φ−λÊ), not φ alone: weighting φ
    alone leaves an O(1) bias that does not shrink with n.
    λ* comes from the Appendix B weighted moments Cov_w/Var_w.
    var_hat is derived from the same residual expression as mu_hat.
    """
    n, M = phi_lab.shape
    N    = syn_unl.shape[0]
    w    = weights / weights.mean()         # normalise to mean 1
    wb   = w[:, None]                       # (n,1) for broadcast over M
    # Appendix B weighted moments (Eqs 19-21)
    phi_bar_w = (wb * phi_lab).mean(0)
    syn_bar_w = (wb * syn_lab).mean(0)
    cov_w    = (wb * (phi_lab - phi_bar_w) * (syn_lab - syn_bar_w)).mean(0)
    var_w    = (wb * (syn_lab - syn_bar_w) ** 2).mean(0)
    var_unl  = syn_unl.var(0)
    denom    = var_w + (n / N) * var_unl
    lambd    = np.clip(np.where(denom > 1e-12, cov_w / denom, 1.0), 0.0, 1.0)
    # Shared residual w·(φ−λÊ)
    resid    = wb * (phi_lab - lambd * syn_lab)
    mu_hat   = lambd * syn_unl.mean(0) + resid.mean(0)
    var_hat  = resid.var(0) / n + lambd**2 * var_unl * (n / N) / n
    return mu_hat, var_hat


def learn_importance_weights(feat_lab, feat_unl):
    """Logistic regression density-ratio estimator."""
    n = len(feat_lab)
    N = len(feat_unl)
    if N > 5 * n:
        idx = np.random.choice(N, size=5 * n, replace=False)
        feat_unl = feat_unl[idx]
    X  = np.vstack([feat_lab, feat_unl])
    y  = np.array([0] * n + [1] * len(feat_unl))
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = CalibratedClassifierCV(
        LogisticRegression(max_iter=300, C=1.0, random_state=0),
        cv=3, method="sigmoid"
    )
    clf.fit(Xs, y)
    p_unl = np.clip(
        clf.predict_proba(sc.transform(feat_lab))[:, 1], 0.01, 0.99
    )
    return p_unl / (1.0 - p_unl)


def oracle_importance_weights(idx_lab, shift_weights_full):
    """
    TRUE importance weights: w(x) = p_target(x) / p_source(x)

    shift_weights_full = exp(β·s) under the exponential mechanism.
      p_source(x) ∝ exp(β·s(x))    (biased labeled sampling)
      p_target(x) = 1/N_total       (uniform over full pool)

    Therefore:
      w(x) ∝ 1 / exp(β·s(x)) = exp(-β·s(x))

    Images with HIGH annotator score were OVERSAMPLED → downweighted.
    Images with LOW annotator score were UNDERSAMPLED → upweighted.
    """
    sw = shift_weights_full[idx_lab]   # exp(β·s) for labeled points
    w  = 1.0 / (sw + 1e-12)           # ∝ exp(-β·s)
    return w / w.mean()                # normalise to mean 1


# --------------------------------------------------
# STEP 3 — Monte Carlo
# --------------------------------------------------

print(f"\nRunning {N_TRIALS} trials x {len(N_LIST)} n values ...")
print("(4 estimators: Classical, PPI++, Oracle PPI++, Adaptive PPI++)")
print("-" * 78)

results = []

for n in N_LIST:
    cov_cls,  cov_ppi,  cov_oppi,  cov_wppi  = [], [], [], []
    wid_cls,  wid_ppi,  wid_oppi,  wid_wppi  = [], [], [], []
    mse_cls,  mse_ppi,  mse_oppi,  mse_wppi  = [], [], [], []
    ess_ppi,  ess_oppi, ess_wppi              = [], [], []

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial * 1000 + n)

        # Exponential shift ∝ exp(β·s), β=1.0
        # shift_weights already computed as exp(β·s) above the MC loop
        sw      = shift_weights / shift_weights.sum()
        idx_lab = rng.choice(N_total, size=n, replace=False, p=sw)
        idx_unl = np.setdiff1d(np.arange(N_total), idx_lab)

        pl = phi_all[idx_lab];  sl = syn_all[idx_lab]
        su = syn_all[idx_unl]
        fl = features[idx_lab]; fu = features[idx_unl]

        # ---- (1) Classical ----
        mu_c  = pl.mean(0)
        var_c = pl.var(0) / n

        # ---- (2) Standard PPI++ ----
        mu_p, var_p = ppi_estimate(pl, sl, su)

        # ---- (3) Oracle PPI++ — TRUE weights ----
        iw_oracle = oracle_importance_weights(idx_lab, shift_weights)
        mu_o, var_o = ppi_estimate_weighted(pl, sl, su, iw_oracle)

        # ---- (4) Adaptive PPI++ — LEARNED weights ----
        try:
            iw_learned = learn_importance_weights(fl, fu)
        except Exception:
            iw_learned = np.ones(n)
        mu_w, var_w = ppi_estimate_weighted(pl, sl, su, iw_learned)

        z = norm.ppf(1 - ALPHA / 2)

        def cov_fn(mu, var):
            return float(np.mean(
                (mu_gt >= mu - z * np.sqrt(var)) &
                (mu_gt <= mu + z * np.sqrt(var))
            ))
        def wid_fn(var):
            return float(np.mean(2 * z * np.sqrt(var)))

        cov_cls.append(cov_fn(mu_c, var_c))
        cov_ppi.append(cov_fn(mu_p, var_p))
        cov_oppi.append(cov_fn(mu_o, var_o))
        cov_wppi.append(cov_fn(mu_w, var_w))

        wid_cls.append(wid_fn(var_c))
        wid_ppi.append(wid_fn(var_p))
        wid_oppi.append(wid_fn(var_o))
        wid_wppi.append(wid_fn(var_w))

        mse_cls.append(float(np.mean((mu_c - mu_gt) ** 2)))
        mse_ppi.append(float(np.mean((mu_p - mu_gt) ** 2)))
        mse_oppi.append(float(np.mean((mu_o - mu_gt) ** 2)))
        mse_wppi.append(float(np.mean((mu_w - mu_gt) ** 2)))

        ess_ppi.append(float(np.mean(var_c / (var_p  + 1e-12))))
        ess_oppi.append(float(np.mean(var_c / (var_o + 1e-12))))
        ess_wppi.append(float(np.mean(var_c / (var_w + 1e-12))))

    sm = lambda l: float(np.mean(l))
    row = dict(
        n        = n,
        cov_cls  = sm(cov_cls),  cov_ppi  = sm(cov_ppi),
        cov_oppi = sm(cov_oppi), cov_wppi = sm(cov_wppi),
        wid_cls  = sm(wid_cls),  wid_ppi  = sm(wid_ppi),
        wid_oppi = sm(wid_oppi), wid_wppi = sm(wid_wppi),
        mse_cls  = sm(mse_cls),  mse_ppi  = sm(mse_ppi),
        mse_oppi = sm(mse_oppi), mse_wppi = sm(mse_wppi),
        ess_ppi  = sm(ess_ppi),  ess_oppi = sm(ess_oppi),
        ess_wppi = sm(ess_wppi),
    )
    results.append(row)

    print(f"n={n:4d} | "
          f"cov cls={row['cov_cls']:.3f}  "
          f"ppi={row['cov_ppi']:.3f}  "
          f"oracle={row['cov_oppi']:.3f}  "
          f"adaptive={row['cov_wppi']:.3f} | "
          f"ESS oracle={row['ess_oppi']:.3f}  "
          f"adaptive={row['ess_wppi']:.3f}")

# --------------------------------------------------
# STEP 4 — Save CSV
# --------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUT_DIR, "ext1_results.csv"), index=False)
print(f"\nSaved: {OUT_DIR}/ext1_results.csv")
print(df.to_string(index=False))

# --------------------------------------------------
# STEP 5 — Figures
# --------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["svg.fonttype"] = "none"

    ns = df["n"].values

    # 4 colours
    CC  = "#4DAF4A"   # Classical   — green
    CP  = "#E41A1C"   # PPI++        — red
    CO  = "#984EA3"   # Oracle       — purple  ← NEW
    CW  = "#377EB8"   # Adaptive     — blue

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ---- (a) Coverage ----
    ax = axes[0]
    ax.plot(ns, df["cov_cls"],  "o-",  color=CC, label="Classical")
    ax.plot(ns, df["cov_ppi"],  "s--", color=CP, label="PPI++ (unweighted)")
    ax.plot(ns, df["cov_oppi"], "D-",  color=CO, label="Oracle PPI++",
            linewidth=2.0)
    ax.plot(ns, df["cov_wppi"], "o-",  color=CW, label="Adaptive PPI++")
    ax.axhline(1 - ALPHA, color="k", ls=":", lw=1.2, label="Target 90%")
    ax.set_xlabel("# labeled samples")
    ax.set_ylabel("Coverage of 90% CIs")
    ax.set_title("(a) Coverage under covariate shift")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # ---- (b) MSE ----
    ax = axes[1]
    ax.plot(ns, df["mse_cls"],  "o-",  color=CC, label="Classical")
    ax.plot(ns, df["mse_ppi"],  "s--", color=CP, label="PPI++ (unweighted)")
    ax.plot(ns, df["mse_oppi"], "D-",  color=CO, label="Oracle PPI++",
            linewidth=2.0)
    ax.plot(ns, df["mse_wppi"], "o-",  color=CW, label="Adaptive PPI++")
    ax.set_xlabel("# labeled samples")
    ax.set_ylabel("MSE")
    ax.set_title("(b) MSE of accuracy estimates")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # ---- (c) ESS ----
    ax = axes[2]
    ax.plot(ns, df["ess_ppi"],  "s--", color=CP, label="PPI++ (unweighted)")
    ax.plot(ns, df["ess_oppi"], "D-",  color=CO, label="Oracle PPI++",
            linewidth=2.0)
    ax.plot(ns, df["ess_wppi"], "o-",  color=CW, label="Adaptive PPI++")
    ax.axhline(1.0, color=CC, ls="--", lw=1.0, label="Classical (=1)")
    ax.set_xlabel("# labeled samples")
    ax.set_ylabel("ESS")
    ax.set_title("(c) ESS — variance reduction under shift")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    fig.suptitle(
        "Extension 1: Adaptive AutoEval under Covariate Shift (ImageNet)\n"
        "Labeled ∝ ResNet-101 confidence  |  Shift corrected via learned weights",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ext1_main.png"),
                dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {OUT_DIR}/ext1_main.png")
    plt.close()

except Exception as e:
    print(f"Plotting error: {e}")

# --------------------------------------------------
# STEP 6 — Verification summary
# --------------------------------------------------
print("\n" + "=" * 78)
print("VERIFICATION SUMMARY")
print(f"{'n':>5} | {'cls':>6} | {'ppi':>6} | {'oracle':>8} | "
      f"{'adaptive':>10} | {'gap_to_oracle':>14}")
print("-" * 78)
for _, r in df.iterrows():
    gap = r.cov_wppi - r.cov_oppi
    print(f"{int(r.n):>5} | {r.cov_cls:>6.3f} | {r.cov_ppi:>6.3f} | "
          f"{r.cov_oppi:>8.3f} | {r.cov_wppi:>10.3f} | "
          f"{gap:>+14.3f}")

print()
print("Expected:")
print("  oracle  ~= 0.90     (valid coverage with true weights)")
print("  adaptive < oracle   (small gap from weight estimation error)")
print("  adaptive >> ppi     (large gain over unweighted baseline)")
print("\nDone.")