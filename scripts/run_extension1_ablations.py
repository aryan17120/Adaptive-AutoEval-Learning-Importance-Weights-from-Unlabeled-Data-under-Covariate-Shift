"""
Extension 1: Ablation Studies for Adaptive AutoEval
Experiments on ImageNet (ResNet18-152, biased sampling ∝ ResNet-101)

Three ablations:
  A1 — Classifier capacity: LogReg vs MLP vs RandomForest
  A2 — Weight clipping threshold: ε ∈ {0.001, 0.01, 0.05, 0.1, 0.2}
  A3 — Shift severity: β ∈ {0.0, 0.5, 1.0, 2.0, 3.0}

Outputs:
  results/ablations/abl_classifier.csv / .png
  results/ablations/abl_clipping.csv   / .png
  results/ablations/abl_shift.csv      / .png
  results/ablations/abl_combined.png   (single figure for paper)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
PHI_DIR = "results/phi_imagenet"
SYN_DIR = "results/synthetic_imagenet"
OUT_DIR = "results/ablations"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
N_TRIALS    = 250
ALPHA       = 0.1
SEED        = 42
np.random.seed(SEED)

# --------------------------------------------------
# Load data
# --------------------------------------------------
print("=" * 74)
print("Extension 1: Ablation Studies")
print("=" * 74)
print("\nLoading ImageNet data...")
phi_all = np.stack(
    [np.load(os.path.join(PHI_DIR, f"{m}.npy")) for m in MODEL_NAMES], axis=1
)
syn_all = np.stack(
    [np.load(os.path.join(SYN_DIR, f"{m}.npy")) for m in MODEL_NAMES], axis=1
)
N_total, M = phi_all.shape
mu_gt    = phi_all.mean(axis=0)

# Weight-estimation features, independent of the shift variable.
# ResNet-50 penultimate features if available; else the entropy proxy.
FEAT_PATH = "results/features_imagenet/resnet50_penultimate.npy"
if os.path.exists(FEAT_PATH):
    features = np.load(FEAT_PATH)
    FEATURE_SOURCE = "resnet50_penultimate_2048d"
else:
    p_syn = np.clip(syn_all, 1e-7, 1.0)
    p_syn = p_syn / p_syn.sum(axis=1, keepdims=True)
    features = -np.sum(p_syn * np.log(p_syn), axis=1, keepdims=True)
    FEATURE_SOURCE = "softmax_entropy_proxy_1d"
    print(f"  Using feature proxy: {FEATURE_SOURCE}")
print(f"Loaded: {N_total} images, {M} models")
print(f"Ground-truth accuracies: {mu_gt}")

# --------------------------------------------------
# Core estimators (same as main experiment)
# --------------------------------------------------
def ppi_estimate(phi_lab, syn_lab, syn_unl):
    n, M = phi_lab.shape; N = syn_unl.shape[0]
    cov_num  = np.mean((phi_lab - phi_lab.mean(0)) *
                       (syn_lab - syn_lab.mean(0)), axis=0)
    var_full = (n / N) * syn_unl.var(0) + syn_lab.var(0)
    lambd    = np.clip(np.where(var_full > 1e-12, cov_num / var_full, 1.0), 0.0, 1.0)
    mu_hat   = lambd * syn_unl.mean(0) + (phi_lab - lambd * syn_lab).mean(0)
    resid    = phi_lab - lambd * syn_lab
    var_hat  = resid.var(0) / n + lambd**2 * syn_unl.var(0) * (n / N) / n
    return mu_hat, var_hat


def ppi_estimate_weighted(phi_lab, syn_lab, syn_unl, weights):
    # Weight the full residual (φ−λÊ); λ* from the Appendix B weighted
    # moments; var_hat from the same residual as mu_hat.
    n, M = phi_lab.shape; N = syn_unl.shape[0]
    w    = weights / weights.mean()
    wb   = w[:, None]
    # Appendix B weighted moments
    phi_bar_w = (wb * phi_lab).mean(0)
    syn_bar_w = (wb * syn_lab).mean(0)
    cov_w    = (wb * (phi_lab - phi_bar_w) * (syn_lab - syn_bar_w)).mean(0)
    var_w    = (wb * (syn_lab - syn_bar_w) ** 2).mean(0)
    var_unl  = syn_unl.var(0)
    denom    = var_w + (n / N) * var_unl
    lambd    = np.clip(np.where(denom > 1e-12, cov_w / denom, 1.0), 0.0, 1.0)
    # Shared residual
    resid    = wb * (phi_lab - lambd * syn_lab)
    mu_hat   = lambd * syn_unl.mean(0) + resid.mean(0)
    var_hat  = resid.var(0) / n + lambd**2 * var_unl * (n / N) / n
    return mu_hat, var_hat


def learn_weights(feat_lab, feat_unl, classifier="logreg",
                  clip_lo=0.01, clip_hi=0.99):
    """
    General importance weight learner.
    classifier: one of 'logreg', 'mlp', 'rf'
    clip_lo/hi: probability clipping bounds
    """
    n = len(feat_lab); N = len(feat_unl)
    if N > 5 * n:
        idx = np.random.choice(N, size=5 * n, replace=False)
        feat_unl = feat_unl[idx]
    X = np.vstack([feat_lab, feat_unl])
    y = np.array([0]*n + [1]*len(feat_unl))
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    if classifier == "logreg":
        base = LogisticRegression(max_iter=300, C=1.0, random_state=0)
        clf  = CalibratedClassifierCV(base, cv=3, method="sigmoid")

    elif classifier == "mlp":
        base = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            max_iter=200,
            random_state=0,
            early_stopping=True,
            validation_fraction=0.1,
        )
        clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")

    elif classifier == "rf":
        base = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=0,
        )
        clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")

    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    clf.fit(Xs, y)
    p_unl = np.clip(clf.predict_proba(sc.transform(feat_lab))[:, 1],
                    clip_lo, clip_hi)
    w = p_unl / (1.0 - p_unl)
    return w / w.mean()


def run_trials(N_LIST, shift_beta, n_trials, classifier="logreg",
               clip_lo=0.01, clip_hi=0.99):
    """
    Run Monte Carlo trials for given shift strength and classifier.
    Returns DataFrame with coverage, MSE, ESS per n.
    """
    # Compute shifted sampling weights
    shift_weights = syn_all[:, 3]        # ResNet-101 drives shift
    sw = np.exp(shift_beta * shift_weights)
    sw = sw / sw.sum()

    results = []
    for n in N_LIST:
        cov_wppi, mse_wppi, ess_wppi = [], [], []
        cov_cls,  mse_cls           = [], []

        for trial in range(n_trials):
            rng = np.random.RandomState(trial * 1000 + n)

            idx_lab = rng.choice(N_total, size=n, replace=False, p=sw)
            idx_unl = np.setdiff1d(np.arange(N_total), idx_lab)

            pl = phi_all[idx_lab]; sl = syn_all[idx_lab]
            su = syn_all[idx_unl]
            fl = features[idx_lab]; fu = features[idx_unl]

            # Classical
            mu_c  = pl.mean(0); var_c = pl.var(0) / n

            # Weighted PPI++
            try:
                iw = learn_weights(fl, fu, classifier=classifier,
                                   clip_lo=clip_lo, clip_hi=clip_hi)
            except Exception:
                iw = np.ones(n)
            mu_w, var_w = ppi_estimate_weighted(pl, sl, su, iw)

            z = norm.ppf(1 - ALPHA / 2)
            cov_fn = lambda mu, var: float(
                np.mean((mu_gt >= mu - z*np.sqrt(var)) &
                        (mu_gt <= mu + z*np.sqrt(var))))

            cov_cls.append(cov_fn(mu_c, var_c))
            cov_wppi.append(cov_fn(mu_w, var_w))
            mse_cls.append(float(np.mean((mu_c - mu_gt)**2)))
            mse_wppi.append(float(np.mean((mu_w - mu_gt)**2)))
            ess_wppi.append(float(np.mean(var_c / (var_w + 1e-12))))

        sm = lambda l: float(np.mean(l))
        results.append(dict(n=n,
                            cov_cls=sm(cov_cls),   cov_wppi=sm(cov_wppi),
                            mse_cls=sm(mse_cls),   mse_wppi=sm(mse_wppi),
                            ess_wppi=sm(ess_wppi)))
    return pd.DataFrame(results)


# ============================================================
# ABLATION 1: Classifier Capacity
# ============================================================
print("\n" + "=" * 74)
print("ABLATION 1: Classifier Capacity")
print("=" * 74)

N_LIST_A1  = [50, 100, 200, 300, 400, 500]
CLASSIFIERS = {
    "Logistic Regression": "logreg",
    "MLP (64×64)":         "mlp",
    "Random Forest":       "rf",
}

abl1_results = {}
for name, clf_key in CLASSIFIERS.items():
    print(f"\n  Running: {name}...")
    df_clf = run_trials(N_LIST_A1, shift_beta=1.0,
                        n_trials=N_TRIALS, classifier=clf_key)
    abl1_results[name] = df_clf
    print(f"  Coverage: "
          + "  ".join([f"n={r.n}: {r.cov_wppi:.3f}"
                       for _, r in df_clf.iterrows()]))

# Save
for name, df_clf in abl1_results.items():
    df_clf.to_csv(os.path.join(OUT_DIR, f"abl_clf_{name.replace(' ', '_').replace('(','').replace(')','')}.csv"),
                 index=False)
print("\nAblation 1 complete.")

# ============================================================
# ABLATION 2: Weight Clipping Threshold
# ============================================================
print("\n" + "=" * 74)
print("ABLATION 2: Weight Clipping Threshold")
print("=" * 74)

N_LIST_A2   = [50, 100, 200, 300, 400, 500]
CLIP_VALUES = [0.001, 0.01, 0.05, 0.10, 0.20]

abl2_results = {}
for eps in CLIP_VALUES:
    print(f"\n  Running: clip ε = {eps}...")
    df_clip = run_trials(N_LIST_A2, shift_beta=1.0,
                         n_trials=N_TRIALS, classifier="logreg",
                         clip_lo=eps, clip_hi=1-eps)
    abl2_results[eps] = df_clip
    print(f"  Coverage: "
          + "  ".join([f"n={r.n}: {r.cov_wppi:.3f}"
                       for _, r in df_clip.iterrows()]))

for eps, df_clip in abl2_results.items():
    df_clip.to_csv(os.path.join(OUT_DIR, f"abl_clip_{str(eps).replace('.','p')}.csv"),
                  index=False)
print("\nAblation 2 complete.")

# ============================================================
# ABLATION 3: Shift Severity
# ============================================================
print("\n" + "=" * 74)
print("ABLATION 3: Shift Severity (Support Overlap)")
print("=" * 74)

N_LIST_A3  = [50, 100, 200, 300, 400, 500]
BETA_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0]

abl3_results = {}
for beta in BETA_VALUES:
    print(f"\n  Running: β = {beta}...")
    df_beta = run_trials(N_LIST_A3, shift_beta=beta,
                         n_trials=N_TRIALS, classifier="logreg")
    abl3_results[beta] = df_beta
    cov_row = df_beta[df_beta.n == 200].iloc[0]
    print(f"  At n=200: cov_cls={cov_row.cov_cls:.3f}  cov_wppi={cov_row.cov_wppi:.3f}")

for beta, df_beta in abl3_results.items():
    df_beta.to_csv(os.path.join(OUT_DIR,
                  f"abl_beta_{str(beta).replace('.','p')}.csv"), index=False)
print("\nAblation 3 complete.")

# ============================================================
# COMBINED FIGURE (paper-ready 3-panel)
# ============================================================
print("\nGenerating combined ablation figure...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ns = np.array(N_LIST_A1)

# Colour palettes
COLORS_A1 = {"Logistic Regression": "#377EB8",
             "MLP (64×64)":         "#E41A1C",
             "Random Forest":       "#FF7F00"}
COLORS_A2  = plt.cm.viridis(np.linspace(0.1, 0.9, len(CLIP_VALUES)))
COLORS_A3  = plt.cm.plasma(np.linspace(0.1, 0.9, len(BETA_VALUES)))

# ---- Panel (a): Classifier capacity ----
ax = axes[0]
for name, df_clf in abl1_results.items():
    ax.plot(df_clf["n"], df_clf["cov_wppi"], "o-",
            color=COLORS_A1[name], label=name, linewidth=2)
ax.axhline(1 - ALPHA, color="k", ls=":", lw=1.2, label="Target 90%")
ax.set_xlabel("# labeled samples"); ax.set_ylabel("Coverage of 90% CIs")
ax.set_title("(a) Effect of classifier capacity")
ax.set_ylim(0.0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- Panel (b): Clipping threshold ----
ax = axes[1]
for i, (eps, df_clip) in enumerate(abl2_results.items()):
    ax.plot(df_clip["n"], df_clip["cov_wppi"], "o-",
            color=COLORS_A2[i], label=f"ε = {eps}", linewidth=2)
ax.axhline(1 - ALPHA, color="k", ls=":", lw=1.2, label="Target 90%")
ax.set_xlabel("# labeled samples"); ax.set_ylabel("Coverage of 90% CIs")
ax.set_title("(b) Sensitivity to clipping threshold ε")
ax.set_ylim(0.0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- Panel (c): Shift severity ----
ax = axes[2]
for i, (beta, df_beta) in enumerate(abl3_results.items()):
    ls = "-" if beta == 1.0 else "--"
    lw = 2.5 if beta == 1.0 else 1.5
    ax.plot(df_beta["n"], df_beta["cov_wppi"], "o",
            color=COLORS_A3[i], linestyle=ls, linewidth=lw,
            label=f"β = {beta}")
ax.axhline(1 - ALPHA, color="k", ls=":", lw=1.2, label="Target 90%")
ax.set_xlabel("# labeled samples"); ax.set_ylabel("Coverage of 90% CIs")
ax.set_title("(c) Robustness to shift severity β")
ax.set_ylim(0.0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(
    "Ablation Studies: Adaptive AutoEval on ImageNet (ResNet-101 shift)\n"
    "(a) Classifier capacity    (b) Weight clipping    (c) Shift severity",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "abl_combined.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved: {OUT_DIR}/abl_combined.png")

# Individual figures (for appendix)
for panel, (title, df_dict, colors) in enumerate([
    ("Classifier Capacity",   abl1_results, COLORS_A1),
    ("Clipping Threshold",    abl2_results, COLORS_A2),
    ("Shift Severity",        abl3_results, COLORS_A3),
]):
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for i, (key, df_k) in enumerate(df_dict.items()):
        c = colors[key] if isinstance(colors, dict) else colors[i]
        ax2.plot(df_k["n"], df_k["cov_wppi"], "o-",
                 color=c, label=str(key), linewidth=2)
    ax2.axhline(1 - ALPHA, color="k", ls=":", lw=1.2, label="Target 90%")
    ax2.set_xlabel("# labeled samples"); ax2.set_ylabel("Coverage of 90% CIs")
    ax2.set_title(f"Ablation: {title}")
    ax2.set_ylim(0.0, 1.05); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"abl_{title.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================
# Summary tables
# ============================================================
print("\n" + "=" * 74)
print("SUMMARY")
print("=" * 74)

print("\n--- Ablation 1: Classifier Capacity (at n=200) ---")
print(f"{'Classifier':25s} | {'cov_wppi':>9} | {'mse_wppi':>9}")
print("-" * 50)
for name, df_clf in abl1_results.items():
    r = df_clf[df_clf.n == 200].iloc[0]
    print(f"{name:25s} | {r.cov_wppi:>9.3f} | {r.mse_wppi:>9.5f}")

print("\n--- Ablation 2: Clipping Threshold (at n=200) ---")
print(f"{'ε':>8} | {'cov_wppi':>9} | {'mse_wppi':>9} | {'ESS':>6}")
print("-" * 45)
for eps, df_clip in abl2_results.items():
    r = df_clip[df_clip.n == 200].iloc[0]
    print(f"{eps:>8.3f} | {r.cov_wppi:>9.3f} | {r.mse_wppi:>9.5f} | {r.ess_wppi:>6.3f}")

print("\n--- Ablation 3: Shift Severity (at n=200) ---")
print(f"{'β':>5} | {'cov_cls':>8} | {'cov_wppi':>9} | {'mse_cls':>8} | {'mse_wppi':>9}")
print("-" * 55)
for beta, df_beta in abl3_results.items():
    r = df_beta[df_beta.n == 200].iloc[0]
    print(f"{beta:>5.1f} | {r.cov_cls:>8.3f} | {r.cov_wppi:>9.3f} | "
          f"{r.mse_cls:>8.5f} | {r.mse_wppi:>9.5f}")

print("\nDone.")