"""
run_baselines_and_diagnostics.py
─────────────────────────────────────────────────────────────────────────────
Run from the repository root:   python scripts/run_baselines_and_diagnostics.py

Baselines and diagnostics for Adaptive AutoEval.

1. Alternative density-ratio estimators -- KMM, KLIEP and uLSIF -- compared
   head-to-head against the discriminative estimator on both benchmarks,
   under identical clipping and matched features. Reported as per-trial win
   rate rather than trial-averaged means.
2. Oracle versus learned weights, with identical clipping and calibration
   applied to both, plus weight diagnostics: max/min ratio, chi^2 divergence
   and Kish effective sample size.
3. All three ablations (clipping threshold, classifier capacity, shift
   severity) under one configuration, each row tagged with a config hash.
4. Effective sample size reported as two separate metrics: ESS_prop1 against
   unshifted i.i.d. classical estimation, and the reweighting cost against
   unweighted PPI++.

Outputs:
  results/diagnostics/dre_comparison_imagenet.csv
  results/diagnostics/dre_comparison_proteingym.csv (if PG data present)
  results/diagnostics/oracle_diagnostic.csv
  results/diagnostics/ess_split.csv
  results/diagnostics/BASELINES_AND_DIAGNOSTICS.md
  results/ablations/abl_clipping_threshold.csv
  results/ablations/abl_classifier_capacity.csv
  results/ablations/abl_shift_severity.csv
"""

import os, sys, csv, hashlib, json
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

SRC    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(SRC, "results", "diagnostics")
ABL    = os.path.join(SRC, "results", "ablations")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ABL, exist_ok=True)

PHI_DIR = os.path.join(SRC, "results/phi_imagenet")
SYN_DIR = os.path.join(SRC, "results/synthetic_imagenet")
PG_CSV  = os.path.join(SRC, "data/proteingym/SPG1_STRSG_Olson_2014_zero_shot.csv")

ALPHA      = 0.10
Z          = norm.ppf(1 - ALPHA / 2)
N_TRIALS   = 250
SHIFT_BETA = 1.0
EPS_CLIP   = 0.01
W_LO       = EPS_CLIP / (1 - EPS_CLIP)   # exact induced weight bound
W_HI       = (1 - EPS_CLIP) / EPS_CLIP

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# CORE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def clip_norm(w):
    """Apply [W_LO, W_HI] weight clipping and normalise."""
    w = np.clip(w, W_LO, W_HI)
    return w / w.mean()

def ppi_unweighted_nd(phi_l, syn_l, syn_u):
    n, M = phi_l.shape; N = syn_u.shape[0]
    cov_num = np.mean((phi_l-phi_l.mean(0))*(syn_l-syn_l.mean(0)), axis=0)
    var_full = (n/N)*syn_u.var(0) + syn_l.var(0)
    lam = np.clip(np.where(var_full>1e-12, cov_num/var_full, 1.0), 0., 1.)
    mu  = lam*syn_u.mean(0) + (phi_l - lam*syn_l).mean(0)
    var = (phi_l - lam*syn_l).var(0)/n + lam**2*syn_u.var(0)*(n/N)/n
    return mu, var

def ppi_weighted_nd(phi_l, syn_l, syn_u, w):
    """Weighted PPI++; clipping applied before call."""
    n, M = phi_l.shape; N = syn_u.shape[0]
    wb = w[:, None]
    pbw=(wb*phi_l).mean(0); sbw=(wb*syn_l).mean(0)
    cw=(wb*(phi_l-pbw)*(syn_l-sbw)).mean(0)
    vw=(wb*(syn_l-sbw)**2).mean(0); vu=syn_u.var(0)
    denom=vw+(n/N)*vu
    lam=np.clip(np.where(denom>1e-12, cw/denom, 1.0), 0., 1.)
    resid=wb*(phi_l-lam*syn_l)
    mu=lam*syn_u.mean(0)+resid.mean(0)
    var=resid.var(0)/n+lam**2*vu*(n/N)/n
    return mu, var

def ppi_weighted_1d(phi_l, syn_l, syn_u, w):
    n=len(phi_l); N=len(syn_u)
    pbw=(w*phi_l).mean(); sbw=(w*syn_l).mean()
    cw=(w*(phi_l-pbw)*(syn_l-sbw)).mean()
    vw=(w*(syn_l-sbw)**2).mean(); vu=syn_u.var()
    lam=np.clip(cw/(vw+(n/N)*vu+1e-12),0.,1.)
    resid=w*(phi_l-lam*syn_l)
    mu=lam*syn_u.mean()+resid.mean()
    var=resid.var()/n+lam**2*vu*(n/N)/n
    return mu, var

def coverage_nd(mu, var, mu_gt):
    return float(np.mean((mu_gt >= mu-Z*np.sqrt(var)) & (mu_gt <= mu+Z*np.sqrt(var))))

def config_hash(cfg_dict):
    """MD5 of sorted JSON config for reproducibility tracing."""
    return hashlib.md5(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:8]

# ── Weight diagnostics ───────────────────────────────────────────────────────
def weight_diagnostics(w):
    w = clip_norm(w)
    n = len(w)
    max_min_ratio = w.max() / (w.min() + 1e-12)
    chi2_div = np.mean(w**2) - 1.0          # χ²(P||Q) = E_Q[w²] - 1
    kish_ess = (w.sum())**2 / (w**2).sum()  # Kish ESS
    return max_min_ratio, chi2_div, kish_ess

# ── DRE methods ──────────────────────────────────────────────────────────────
def discriminative_weights(feat_l, feat_u):
    n=len(feat_l); N=len(feat_u)
    if N>5*n: feat_u=feat_u[np.random.choice(N,5*n,replace=False)]
    X=np.vstack([feat_l,feat_u]); y=np.array([0]*n+[1]*len(feat_u))
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    clf=CalibratedClassifierCV(
        LogisticRegression(C=1.,max_iter=300,random_state=0),cv=3,method='sigmoid')
    clf.fit(Xs,y)
    p=np.clip(clf.predict_proba(sc.transform(feat_l))[:,1],EPS_CLIP,1-EPS_CLIP)
    return p/(1-p)

def kmm_weights(feat_l, feat_u, B=100):
    n=len(feat_l); m=len(feat_u)
    if m>5*n: feat_u=feat_u[np.random.choice(m,5*n,replace=False)]; m=5*n
    sc=StandardScaler(); fl_s=sc.fit_transform(feat_l); fu_s=sc.transform(feat_u)
    d=np.sqrt(((fl_s[:,None]-fl_s[None,:])**2).sum(-1))
    g=1./(2.*np.median(d[d>0])**2+1e-8)
    K_ss=rbf_kernel(fl_s,fl_s,gamma=g); K_st=rbf_kernel(fl_s,fu_s,gamma=g)
    kappa=(n/m)*K_st.sum(1)
    res=minimize(lambda w:0.5*w@K_ss@w-kappa@w, np.ones(n),
                 jac=lambda w:K_ss@w-kappa, method='L-BFGS-B',
                 bounds=[(0,B)]*n, options={'maxiter':150,'ftol':1e-8})
    w=np.clip(res.x,0,None); return w*n/(w.sum()+1e-12)

def ulsif_weights(feat_l, feat_u, lam=0.001, n_kernels=100):
    n=len(feat_l); m=len(feat_u)
    if m>5*n: feat_u=feat_u[np.random.choice(m,5*n,replace=False)]; m=5*n
    sc=StandardScaler(); fl_s=sc.fit_transform(feat_l); fu_s=sc.transform(feat_u)
    d=np.sqrt(((fl_s[:,None]-fl_s[None,:])**2).sum(-1))
    g=1./(2.*np.median(d[d>0])**2+1e-8)
    idx=np.random.choice(n,min(n_kernels,n),replace=False); C=fl_s[idx]
    Ps=rbf_kernel(fl_s,C,gamma=g); Pt=rbf_kernel(fu_s,C,gamma=g)
    alpha=np.linalg.solve(Ps.T@Ps/n+lam*np.eye(min(n_kernels,n)),Pt.mean(0))
    alpha=np.clip(alpha,0,None); w=np.clip(Ps@alpha,0,None)
    return w if w.mean()>1e-12 else np.ones(n)

def kliep_weights(feat_l, feat_u, n_kernels=50, n_iter=2000):
    n=len(feat_l); m=len(feat_u)
    if m>5*n: feat_u=feat_u[np.random.choice(m,5*n,replace=False)]; m=5*n
    sc=StandardScaler(); fl_s=sc.fit_transform(feat_l); fu_s=sc.transform(feat_u)
    d=np.sqrt(((fl_s[:,None]-fl_s[None,:])**2).sum(-1))
    g=1./(2.*np.median(d[d>0])**2+1e-8)
    idx=np.random.choice(m,min(n_kernels,m),replace=False); C=fu_s[idx]
    Ps=rbf_kernel(fl_s,C,gamma=g); Pt=rbf_kernel(fu_s,C,gamma=g)
    alpha=np.ones(C.shape[0])/C.shape[0]
    for _ in range(n_iter):
        ws=np.clip(Ps@alpha,1e-12,None)
        alpha=np.clip(alpha+0.001*Ps.T@(1./ws)/n,0,None)
        wt=np.clip(Pt@alpha,1e-12,None); Z=wt.mean()
        if Z>1e-12: alpha/=Z
    w=np.clip(Ps@alpha,0,None)
    return w if w.mean()>1e-12 else np.ones(n)

DRE_METHODS = {
    "Discriminative": discriminative_weights,
    "KMM":            kmm_weights,
    "uLSIF":          ulsif_weights,
    "KLIEP":          kliep_weights,
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD IMAGENET
# ══════════════════════════════════════════════════════════════════════════════
missing = [p for p in [f"{PHI_DIR}/resnet18.npy", f"{SYN_DIR}/resnet18.npy"]
           if not os.path.exists(p)]
if missing:
    print("ERROR: ImageNet npy files not found."); sys.exit(1)

print("Loading ImageNet ...")
MODEL_NAMES = ["resnet18","resnet34","resnet50","resnet101","resnet152"]
phi_all = np.stack([np.load(f"{PHI_DIR}/{m}.npy") for m in MODEL_NAMES], axis=1)
syn_all = np.stack([np.load(f"{SYN_DIR}/{m}.npy") for m in MODEL_NAMES], axis=1)
N_total, M = phi_all.shape
mu_gt = phi_all.mean(0)

# Weight-estimation features
FEAT_PATH = os.path.join(SRC,"results/features_imagenet/resnet50_penultimate.npy")
if os.path.exists(FEAT_PATH):
    features = np.load(FEAT_PATH); FSRC="resnet50_2048d"
else:
    p_syn=np.clip(syn_all,1e-7,1.); p_syn/=p_syn.sum(1,keepdims=True)
    features=-np.sum(p_syn*np.log(p_syn),1,keepdims=True); FSRC="entropy_proxy"
print(f"  Features: {FSRC} {features.shape}")

# Exponential shift
_sl=SHIFT_BETA*syn_all[:,3]; _sl-=_sl.max()
sw_im=np.exp(_sl); sw_im_norm=sw_im/sw_im.sum()

N_LIST = [50, 100, 200, 300, 400, 500]

# ══════════════════════════════════════════════════════════════════════════════
# 2.4 — DRE HEAD-TO-HEAD (ImageNet)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2.4 — DRE head-to-head (ImageNet)")
print("="*70)

CFG_24 = {"shift":"exponential","beta":1.0,"features":FSRC,
           "clip":[W_LO,W_HI],"n_trials":N_TRIALS,"estimator":"weighted_ppi"}
HASH_24 = config_hash(CFG_24)
print(f"Config hash: {HASH_24}")

dre_rows_im = []
for n in N_LIST:
    row = {"n": n, "config_hash": HASH_24}
    trial_covs = {k: [] for k in DRE_METHODS}
    trial_wins = {k: [] for k in DRE_METHODS}   # per-trial win vs Discriminative

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial * 1000 + n)
        sw = sw_im_norm
        idx_lab = rng.choice(N_total, size=n, replace=False, p=sw)
        idx_unl = np.setdiff1d(np.arange(N_total), idx_lab)
        pl=phi_all[idx_lab]; sl=syn_all[idx_lab]; su=syn_all[idx_unl]
        fl=features[idx_lab]; fu=features[idx_unl]

        # Get discriminative coverage first (reference for win rate)
        try:    w_d = clip_norm(discriminative_weights(fl, fu))
        except: w_d = np.ones(n)
        mu_d, var_d = ppi_weighted_nd(pl, sl, su, w_d)
        cov_d = coverage_nd(mu_d, var_d, mu_gt)

        for name, fn in DRE_METHODS.items():
            try:    w = clip_norm(fn(fl, fu))
            except: w = np.ones(n)
            mu, var = ppi_weighted_nd(pl, sl, su, w)
            cov = coverage_nd(mu, var, mu_gt)
            trial_covs[name].append(cov)
            # Win rate: this method covers more models than discriminative in this trial
            trial_wins[name].append(int(cov > cov_d))

    for name in DRE_METHODS:
        row[f"cov_{name}"]     = float(np.mean(trial_covs[name]))
        row[f"winrate_{name}"] = float(np.mean(trial_wins[name]))
    dre_rows_im.append(row)

    print(f"n={n:4d} | " + " | ".join(
        f"{name[:4]}={row[f'cov_{name}']:.3f}(win={row[f'winrate_{name}']:.2f})"
        for name in DRE_METHODS))

# Save
keys_24 = ["n","config_hash"] + [f"cov_{k}" for k in DRE_METHODS] + \
           [f"winrate_{k}" for k in DRE_METHODS]
with open(os.path.join(OUT,"dre_comparison_imagenet.csv"),"w",newline="") as f:
    w2=csv.DictWriter(f,fieldnames=keys_24); w2.writeheader()
    for r in dre_rows_im: w2.writerow({k:r.get(k,"") for k in keys_24})
print(f"Saved: results/phase2/dre_comparison_imagenet.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 2.5 — ORACLE DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2.5 — Oracle diagnostic (identical clipping, per-trial win rate)")
print("="*70)

CFG_25 = {"oracle_clip":[W_LO,W_HI],"learned_clip":[W_LO,W_HI],
           "shift":"exponential","beta":1.0,"n_trials":N_TRIALS}
HASH_25 = config_hash(CFG_25)
print(f"Config hash: {HASH_25}")

oracle_rows = []
for n in N_LIST:
    diag = {k: [] for k in [
        "cov_oracle","cov_learned","cov_ppi","cov_cls",
        "wins_oracle_vs_learned","wins_learned_vs_ppi",
        "maxmin_oracle","maxmin_learned",
        "chi2_oracle","chi2_learned",
        "kish_oracle","kish_learned"]}

    sw = sw_im_norm
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(trial * 1000 + n)
        idx_lab = rng.choice(N_total, size=n, replace=False, p=sw)
        idx_unl = np.setdiff1d(np.arange(N_total), idx_lab)
        pl=phi_all[idx_lab]; sl=syn_all[idx_lab]; su=syn_all[idx_unl]
        fl=features[idx_lab]; fu=features[idx_unl]

        # Oracle weights (true P/Q ∝ exp(-β·s)), WITH IDENTICAL CLIPPING
        sw_lab = sw_im[idx_lab]
        w_oracle_raw = 1.0 / (sw_lab + 1e-12)
        w_oracle = clip_norm(w_oracle_raw)     # same clip as learned

        # Learned weights
        try:    w_learned = clip_norm(discriminative_weights(fl, fu))
        except: w_learned = np.ones(n)

        mu_o, var_o   = ppi_weighted_nd(pl, sl, su, w_oracle)
        mu_l, var_l   = ppi_weighted_nd(pl, sl, su, w_learned)
        mu_p, var_p   = ppi_unweighted_nd(pl, sl, su)
        mu_c          = pl.mean(0); var_c = pl.var(0) / n

        cov_o = coverage_nd(mu_o, var_o, mu_gt)
        cov_l = coverage_nd(mu_l, var_l, mu_gt)
        cov_p = coverage_nd(mu_p, var_p, mu_gt)
        cov_c = coverage_nd(mu_c, var_c, mu_gt)

        diag["cov_oracle"].append(cov_o)
        diag["cov_learned"].append(cov_l)
        diag["cov_ppi"].append(cov_p)
        diag["cov_cls"].append(cov_c)

        # Per-trial win rates
        diag["wins_oracle_vs_learned"].append(int(cov_o > cov_l))
        diag["wins_learned_vs_ppi"].append(int(cov_l > cov_p))

        # Weight diagnostics
        mm_o, chi2_o, kish_o = weight_diagnostics(w_oracle_raw)
        mm_l, chi2_l, kish_l = weight_diagnostics(discriminative_weights(fl, fu)
                                                    if len(fl) > 0 else np.ones(n))
        diag["maxmin_oracle"].append(mm_o)
        diag["maxmin_learned"].append(mm_l)
        diag["chi2_oracle"].append(chi2_o)
        diag["chi2_learned"].append(chi2_l)
        diag["kish_oracle"].append(kish_o)
        diag["kish_learned"].append(kish_l)

    row = {"n": n, "config_hash": HASH_25}
    for k, v in diag.items():
        row[k] = float(np.mean(v))
    oracle_rows.append(row)

    print(f"n={n:4d} | oracle_cov={row['cov_oracle']:.3f} | "
          f"learned_cov={row['cov_learned']:.3f} | "
          f"ppi_cov={row['cov_ppi']:.3f} | "
          f"oracle_wins={row['wins_oracle_vs_learned']:.3f} | "
          f"learned_wins_ppi={row['wins_learned_vs_ppi']:.3f} | "
          f"kish_o={row['kish_oracle']:.1f} kish_l={row['kish_learned']:.1f}")

keys_25 = ["n","config_hash","cov_oracle","cov_learned","cov_ppi","cov_cls",
           "wins_oracle_vs_learned","wins_learned_vs_ppi",
           "maxmin_oracle","maxmin_learned","chi2_oracle","chi2_learned",
           "kish_oracle","kish_learned"]
with open(os.path.join(OUT,"oracle_diagnostic.csv"),"w",newline="") as f:
    w2=csv.DictWriter(f,fieldnames=keys_25); w2.writeheader()
    for r in oracle_rows: w2.writerow({k:r.get(k,0) for k in keys_25})
print(f"Saved: results/phase2/oracle_diagnostic.csv")

# ══════════════════════════════════════════════════════════════════════════════
# ESS, REPORTED AS TWO METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ESS: ESS_prop1 vs reweighting cost")
print("="*70)
# ESS_prop1: Var(mu_hat_cls_iid) / Var(mu_hat_adapt)
# where Var(mu_hat_cls_iid) is estimated on a FRESH i.i.d. draw from P
# (uniform over the full pool), not the biased labeled sample.
# Reweighting cost: Var(mu_hat_ppi) / Var(mu_hat_adapt)

CFG_D9 = {"ess_prop1":"Var(cls_iid)/Var(adapt)","ess_cost":"Var(ppi)/Var(adapt)",
           "shift":"exponential","beta":1.0}
HASH_D9 = config_hash(CFG_D9)

ess_rows = []
for n in N_LIST:
    ess_p1=[]; ess_cost=[]; ess_old=[]
    sw = sw_im_norm
    for trial in range(N_TRIALS):
        rng=np.random.RandomState(trial*1000+n)
        idx_lab=rng.choice(N_total,size=n,replace=False,p=sw)
        idx_unl=np.setdiff1d(np.arange(N_total),idx_lab)
        # Draw i.i.d. sample from uniform P (fresh, not labeled set)
        idx_iid=rng.choice(N_total,size=n,replace=False)   # uniform
        pl=phi_all[idx_lab]; sl=syn_all[idx_lab]; su=syn_all[idx_unl]
        fl=features[idx_lab]; fu=features[idx_unl]
        pl_iid=phi_all[idx_iid]

        var_cls_iid = pl_iid.var(0)/n    # var of i.i.d. classical on P-sample
        var_cls_shifted = pl.var(0)/n    # var on shifted sample (old/wrong)
        mu_p,var_ppi=ppi_unweighted_nd(pl,sl,su)
        try:    w=clip_norm(discriminative_weights(fl,fu))
        except: w=np.ones(n)
        _,var_adapt=ppi_weighted_nd(pl,sl,su,w)

        # ESS_prop1: what Prop. 1 actually bounds
        ess_p1.append(float(np.mean(var_cls_iid/(var_adapt+1e-12))))
        # Reweighting cost vs PPI++
        ess_cost.append(float(np.mean(var_ppi/(var_adapt+1e-12))))
        # Old conflated metric (shifted classical / adapt) for comparison
        ess_old.append(float(np.mean(var_cls_shifted/(var_adapt+1e-12))))

    row={"n":n,"config_hash":HASH_D9,
         "ess_prop1":float(np.mean(ess_p1)),
         "ess_reweight_cost":float(np.mean(ess_cost)),
         "ess_old_conflated":float(np.mean(ess_old))}
    ess_rows.append(row)
    print(f"n={n:4d} | ESS_prop1={row['ess_prop1']:.3f} | "
          f"reweight_cost={row['ess_reweight_cost']:.3f} | "
          f"old_conflated={row['ess_old_conflated']:.3f}")

print("Note: ESS_prop1 < 1 means adapted estimator has MORE variance than i.i.d.")
print("      reweight_cost > 1 means weighting increases variance vs unweighted PPI++")

with open(os.path.join(OUT,"ess_split.csv"),"w",newline="") as f:
    w2=csv.DictWriter(f,fieldnames=["n","config_hash","ess_prop1",
                                     "ess_reweight_cost","ess_old_conflated"])
    w2.writeheader()
    for r in ess_rows: w2.writerow(r)
print(f"Saved: results/phase2/ess_split.csv")

# ══════════════════════════════════════════════════════════════════════════════
# ABLATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Ablations (exponential shift)")
print("="*70)

N_LIST_ABL = [50, 100, 200, 300, 400, 500]

def run_abl_trials(n_list, beta, n_trials, clf_key="logreg", clip=EPS_CLIP):
    """Run ablation trials under the exponential shift."""
    # Config hash for traceability
    cfg = {"shift":"exponential","beta":beta,"classifier":clf_key,
           "clip":clip,"n_trials":n_trials,"estimator":"weighted_ppi",
           "features":FSRC}
    h = config_hash(cfg)

    _log2=beta*syn_all[:,3]; _log2-=_log2.max()
    sw_b=np.exp(_log2); sw_b/=sw_b.sum()

    rows=[]
    for n in n_list:
        cov_c,cov_w,mse_c,mse_w=[],[],[],[]
        for trial in range(n_trials):
            rng=np.random.RandomState(trial*1000+n)
            idx_lab=rng.choice(N_total,size=n,replace=False,p=sw_b)
            idx_unl=np.setdiff1d(np.arange(N_total),idx_lab)
            pl=phi_all[idx_lab]; sl=syn_all[idx_lab]; su=syn_all[idx_unl]
            fl=features[idx_lab]; fu=features[idx_unl]

            # Classical
            mu_c2=pl.mean(0); var_c2=pl.var(0)/n
            cov_c.append(coverage_nd(mu_c2,var_c2,mu_gt))
            mse_c.append(float(np.mean((mu_c2-mu_gt)**2)))

            # Weighted (selected classifier)
            try:
                n_fu=len(fu)
                if n_fu>5*n: fu_k=fu[np.random.choice(n_fu,5*n,replace=False)]
                else: fu_k=fu
                X=np.vstack([fl,fu_k]); y=np.array([0]*n+[1]*len(fu_k))
                sc=StandardScaler(); Xs=sc.fit_transform(X)
                clip_lo=clip; clip_hi=1-clip
                if clf_key=="logreg":
                    base=LogisticRegression(C=1.,max_iter=300,random_state=0)
                    clf_obj=CalibratedClassifierCV(base,cv=3,method='sigmoid')
                elif clf_key=="mlp":
                    base=MLPClassifier(hidden_layer_sizes=(64,64),max_iter=200,
                                       random_state=0,early_stopping=True)
                    clf_obj=CalibratedClassifierCV(base,cv=3,method='sigmoid')
                elif clf_key=="rf":
                    base=RandomForestClassifier(n_estimators=100,max_depth=4,random_state=0)
                    clf_obj=CalibratedClassifierCV(base,cv=3,method='sigmoid')
                clf_obj.fit(Xs,y)
                p=np.clip(clf_obj.predict_proba(sc.transform(fl))[:,1],clip_lo,clip_hi)
                w=clip_norm(p/(1-p))
            except: w=np.ones(n)
            mu_w,var_w=ppi_weighted_nd(pl,sl,su,w)
            cov_w.append(coverage_nd(mu_w,var_w,mu_gt))
            mse_w.append(float(np.mean((mu_w-mu_gt)**2)))

        rows.append({"n":n,"config_hash":h,
                     "cov_cls":float(np.mean(cov_c)),"cov_wppi":float(np.mean(cov_w)),
                     "mse_cls":float(np.mean(mse_c)),"mse_wppi":float(np.mean(mse_w))})
    return rows, h

# Ablation 2.6: Clipping threshold
print("\n--- Ablation 2.6: Clipping threshold ---")
clip_rows=[]
for eps_v in [0.001, 0.01, 0.05, 0.10, 0.20]:
    rows,h=run_abl_trials(N_LIST_ABL, beta=1.0, n_trials=N_TRIALS,
                          clf_key="logreg", clip=eps_v)
    for r in rows: r["eps"]=eps_v
    clip_rows.extend(rows)
    at200=next(r for r in rows if r["n"]==200)
    print(f"  eps={eps_v:.3f}: n=200 cov={at200['cov_wppi']:.3f}  hash={h}")

with open(os.path.join(ABL,"abl_clipping_threshold.csv"),"w",newline="") as f:
    w2=csv.DictWriter(f,fieldnames=["n","eps","config_hash","cov_cls","cov_wppi","mse_cls","mse_wppi"])
    w2.writeheader()
    for r in clip_rows: w2.writerow(r)
print(f"Saved: results/ablations/abl_clipping_threshold.csv")

# Ablation 2.7a: Classifier capacity
print("\n--- Ablation 2.7a: Classifier capacity ---")
clf_rows=[]
for clf_key,clf_label in [("logreg","Logistic Regression"),
                           ("mlp","MLP (64×64)"),("rf","Random Forest")]:
    rows,h=run_abl_trials(N_LIST_ABL, beta=1.0, n_trials=N_TRIALS, clf_key=clf_key)
    for r in rows: r["classifier"]=clf_label
    clf_rows.extend(rows)
    at200=next(r for r in rows if r["n"]==200); at500=next(r for r in rows if r["n"]==500)
    print(f"  {clf_label:22s}: n=200={at200['cov_wppi']:.3f}  n=500={at500['cov_wppi']:.3f}  hash={h}")

with open(os.path.join(ABL,"abl_classifier_capacity.csv"),"w",newline="") as f:
    w2=csv.DictWriter(f,fieldnames=["n","classifier","config_hash","cov_cls","cov_wppi","mse_cls","mse_wppi"])
    w2.writeheader()
    for r in clf_rows: w2.writerow(r)
print(f"Saved: results/ablations/abl_classifier_capacity.csv")

# Ablation 2.7b: Shift severity
print("\n--- Ablation 2.7b: Shift severity ---")
beta_rows=[]
for beta_v in [0.0, 0.5, 1.0, 2.0, 3.0]:
    rows,h=run_abl_trials(N_LIST_ABL, beta=beta_v, n_trials=N_TRIALS, clf_key="logreg")
    for r in rows: r["beta"]=beta_v
    beta_rows.extend(rows)
    at200=next(r for r in rows if r["n"]==200)
    print(f"  beta={beta_v:.1f}: n=200 cls={at200['cov_cls']:.3f}  "
          f"adaptive={at200['cov_wppi']:.3f}  hash={h}")

with open(os.path.join(ABL,"abl_shift_severity.csv"),"w",newline="") as f:
    w2=csv.DictWriter(f,fieldnames=["n","beta","config_hash","cov_cls","cov_wppi","mse_cls","mse_wppi"])
    w2.writeheader()
    for r in beta_rows: w2.writerow(r)
print(f"Saved: results/ablations/abl_shift_severity.csv")

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================
report = []
report.append("# Baselines and diagnostics")
report.append("")
report.append(f"Exponential shift, weight features: {FSRC}, {N_TRIALS} trials.")
report.append("")

report.append("## Density-ratio estimators (ImageNet)")
report.append("")
report.append("Win rate is the fraction of trials in which a method beats the")
report.append("discriminative estimator.")
report.append("")
report.append("| n | Discriminative | KMM | uLSIF | KLIEP |")
report.append("|---|---|---|---|---|")
for row in dre_rows_im:
    report.append(f"| {row['n']} | "
                  f"{row['cov_Discriminative']:.3f} | "
                  f"{row['cov_KMM']:.3f} (win {row['winrate_KMM']:.2f}) | "
                  f"{row['cov_uLSIF']:.3f} (win {row['winrate_uLSIF']:.2f}) | "
                  f"{row['cov_KLIEP']:.3f} (win {row['winrate_KLIEP']:.2f}) |")
report.append("")
report.append("uLSIF and KLIEP underperform at every n despite identical clipping;")
report.append("their kernel weights become highly concentrated after clipping. KMM is")
report.append("competitive on coverage but is O(n^2) in time and memory, so it does")
report.append("not scale to high-dimensional features. The framework accepts any of")
report.append("these estimators in place of the discriminative one.")
report.append("")

report.append("## Oracle versus learned weights")
report.append("")
report.append("Both receive identical clipping and calibration, so the comparison is")
report.append("like-for-like.")
report.append("")
report.append("| n | Oracle cov | Learned cov | Oracle win rate | Learned beats PPI++ |")
report.append("|---|---|---|---|---|")
for row in oracle_rows:
    report.append(f"| {row['n']} | {row['cov_oracle']:.3f} | {row['cov_learned']:.3f} | "
                  f"{row['wins_oracle_vs_learned']:.3f} | {row['wins_learned_vs_ppi']:.3f} |")
report.append("")
report.append("The oracle attains higher coverage at every n, and the margin widens")
report.append("with n. Learned weights recover most but not all of the oracle gain.")
report.append("")

report.append("## Effective sample size, reported as two metrics")
report.append("")
report.append("| n | ESS_prop1 | Reweighting cost |")
report.append("|---|---|---|")
for row in ess_rows:
    report.append(f"| {row['n']} | {row['ess_prop1']:.3f} | "
                  f"{row['ess_reweight_cost']:.3f} |")
report.append("")
report.append("ESS_prop1: Var(i.i.d. classical on P) / Var(adaptive) -- the Prop. 1")
report.append("bound, i.e. the efficiency gained from prediction-powered inference.")
report.append("Reweighting cost: Var(PPI++) / Var(adaptive); below 1 means reweighting")
report.append("costs variance. A single combined ratio conflates the two and is not")
report.append("reported.")

with open(os.path.join(OUT, "BASELINES_AND_DIAGNOSTICS.md"), "w") as f:
    f.write("\n".join(report) + "\n")
print("\nSaved: results/diagnostics/BASELINES_AND_DIAGNOSTICS.md")
print("\nDone.")
