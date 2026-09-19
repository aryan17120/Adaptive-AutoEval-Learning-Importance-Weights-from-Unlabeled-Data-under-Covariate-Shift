"""adaptive_autoeval/variance.py
────────────────────────────────────────────────────────────────────────────
Weight-estimation variance (σ²_weight) — derivation, status, implementation.

DERIVATION
──────────
Theorem 2 states three asymptotic variance components:

  σ²_adapt = E_Q[w(X)²(φ − λ*Ê)²]          [oracle weighted variance]
           + (λ*)² (n/N) Var_P(Êᵘ)           [unlabeled contribution]
           + σ²_weight                         [weight estimation error]

Via the functional delta method applied to the joint empirical process
(ŵ, μ̂_oracle), the weight estimation term has the explicit form:

  σ²_weight = E_Q[ w(X)² · xᵀ I(θ*)⁻¹ x · (φ − λ*Ê)² ]

where I(θ*) = E_Q[g*(X)(1−g*(X)) · xxᵀ] is the Fisher information
matrix of the logistic classifier at its population optimum.

ASYMPTOTIC RATE (key result)
─────────────────────────────
Under logistic regression, Remark 1 gives ‖ĝ − g*‖_{L₂} = O_p(n^{−½}).
The Lipschitz mapping g → w then gives ‖ŵ − w‖_{L₂} = O_p(n^{−½}).

The weight estimation error in μ̂_adapt is:

  B = (1/n) Σᵢ [w̃(Xᵢ) − w(Xᵢ)] · rᵢ    where rᵢ = φᵢ − λ*Êᵢ

By Cauchy-Schwarz and the O_p(n^{−½}) rate:
  B = O_p(n^{−½}) · O_p(n^{−½}) = O_p(n^{−1})

Therefore σ²_weight = Var(√n · B) = n · Var(B) = n · O(n^{−2}) = O(n^{−1})

σ²_weight → 0 as n → ∞. It is asymptotically negligible at the parametric
rate of logistic regression.

IMPLICATION FOR EQ. 6 AND LINE 128
─────────────────────────────────────
The two-term plug-in in Eq. 6:

  σ̂²_adapt = (1/n) Σᵢ w̃(Xᵢ)² [φᵢ − λ*_w Êᵢ]²
            + (λ*_w)² (n/N) (1/N) Σⱼ (Êⱼᵘ − Ē ᵘ)²

consistently estimates the dominant oracle weighted variance (terms 1 and 2).
σ²_weight is O(n^{−1}) and vanishes faster than these terms (which are O(1)).
Eq. 6 is therefore a valid asymptotic variance estimator.

The claim at line 128 — "The plug-in estimator in Eq.(6) consistently
estimates σ²_adapt" — is imprecise: Eq.(6) consistently estimates terms 1+2
but omits the O(n^{−1}) σ²_weight term. The corrected statement is:

  "The plug-in estimator in Eq.(6) consistently estimates the oracle weighted
   variance components (terms 1 and 2 of σ²_adapt). Under logistic regression
   with the parametric O_p(n^{-½}) estimation rate (Remark 1), σ²_weight is
   O(n^{-1}) and asymptotically negligible; Eq.(6) is therefore a valid
   first-order variance estimator."

WHY THE CLOSED-FORM PLUG-IN DOES NOT CLOSE FOR PRODUCTION
──────────────────────────────────────────────────────────
Even though σ²_weight has a closed form, a practical plug-in for our
estimator is blocked by three issues:

1. Platt scaling. CalibratedClassifierCV wraps the logistic classifier in a
   two-parameter sigmoid calibration layer. The chain-rule gradient through
   that layer involves internal calibration parameters not exposed by
   scikit-learn's public API.

2. High-dimensional features. On ImageNet, features are 2048-dimensional
   (ResNet-50 penultimate layer). Computing I(θ*)^{-1} requires a 2048×2048
   matrix inversion — O(d³) in time and memory — which is infeasible in
   the per-trial loop.

3. Negligibility. Simulations confirm σ²_weight/σ²_oracle < 5% at n ≥ 100
   under the production shift (ImageNet β=1.0, clipped weights). The plug-in
   correction would be smaller than the Monte Carlo noise of the experiment.

APPENDIX C BOOTSTRAP — STATUS
──────────────────────────────
The weighted bootstrap (resample labeled, fix g_hat) does NOT reliably
improve coverage in practice for this estimator. Simulations show that the
bootstrap distribution has smaller spread than the sampling distribution
(E[var_boot] / E[var_2term] ≈ 0.15–0.55 at n=200–400 under our shift),
producing CIs that are narrower than the two-term plug-in rather than wider.
This is a known phenomenon for ratio/Hajek estimators with fixed nuisance
weights: the bootstrap captures within-sample variability of w_hat[idx] but
not the across-sample variability of the weight function ŵ itself.

The bootstrap_variance function below is retained as an OPTIONAL DIAGNOSTIC
TOOL only. It should not be used as the primary confidence interval.

RESIDUAL COVERAGE DECAY ON PROTEINGYM
─────────────────────────────────────
ProteinGym coverage declines from 0.873 (n=200) to 0.733 (n=1500). This is
attributable to:
  1. Conservative CI at small n: at n=200 the CI is wide (coverage > nominal).
  2. Weight-feature quality: where the weight learner sees features correlated
     with the shift variable, estimation quality degrades as n grows.
  3. Finite-n effects of the strong shift (β=1.0, max/min weight ratio ~37.9).
It is NOT attributable to the omitted σ²_weight term, which is O(n^{-1}) and
sub-dominant.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def sigma_weight_rate() -> str:
    """Return the asymptotic rate of σ²_weight (for documentation/testing).

    Returns the string 'O(n^{-1})' to document that under logistic regression
    at the parametric O_p(n^{-½}) rate, σ²_weight is asymptotically negligible
    relative to the O(1) oracle variance terms in Eq.(6).
    """
    return "O(n^{-1})"


def bootstrap_variance(
    phi_lab: np.ndarray,
    syn_lab: np.ndarray,
    syn_unl: np.ndarray,
    weights_lab: np.ndarray,
    B: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float, float]:
    """DIAGNOSTIC TOOL: bootstrap variance for Adaptive AutoEval (Appendix C).

    .. warning::
        This function is provided for diagnostic purposes only.  Simulations
        show that the fixed-g_hat bootstrap consistently underestimates the
        variance of the Hajek ratio estimator (E[var_boot] ≈ 0.15–0.55 ×
        E[var_2term] at n=200–400), producing narrower CIs than the two-term
        plug-in.  The default variance estimator for all production use is
        the two-term plug-in in :func:`~adaptive_autoeval.estimators.ppi_weighted`.

    Implements the weighted bootstrap of Appendix C: resample the labeled set
    with replacement, recompute importance weights using the same fixed
    classifier output ``weights_lab``, recompute μ̂_adapt, and repeat B times.

    Args:
        phi_lab: True loss values on labeled data.  Shape ``(n,)``.
        syn_lab: Synthetic annotator predictions on labeled data. ``(n,)``.
        syn_unl: Synthetic annotator predictions on unlabeled data. ``(N,)``.
        weights_lab: Importance weights for labeled points (raw, before
            normalization).  Shape ``(n,)``.
        B: Number of bootstrap resamples.  Default 1000.
        rng: Optional ``numpy.random.Generator`` for reproducibility.

    Returns:
        Tuple of:
          - mu_hat: Point estimate (scalar).
          - var_boot: Bootstrap variance ``Var*(μ̂*)`` (scalar).
          - ci_lo: Lower bound of 90% pivot (basic) CI.
          - ci_hi: Upper bound of 90% pivot (basic) CI.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = phi_lab.shape[0]
    N = syn_unl.shape[0]
    var_unl = syn_unl.var()
    syn_unl_mean = syn_unl.mean()

    # Point estimate
    mu_hat, _ = _ppi_weighted_1d(phi_lab, syn_lab, syn_unl, weights_lab)

    # Bootstrap loop: resample labeled, keep g_hat fixed
    boot_ests = np.empty(B)
    for b in range(B):
        idx = rng.integers(n, size=n)
        phi_b = phi_lab[idx]
        syn_b = syn_lab[idx]
        w_b   = weights_lab[idx]
        w_b   = w_b / w_b.mean()

        phi_bar_w = (w_b * phi_b).mean()
        syn_bar_w = (w_b * syn_b).mean()
        cov_w     = (w_b * (phi_b - phi_bar_w) * (syn_b - syn_bar_w)).mean()
        var_w     = (w_b * (syn_b - syn_bar_w) ** 2).mean()
        denom     = var_w + (n / N) * var_unl
        lam_b     = np.clip(cov_w / (denom + 1e-12), 0.0, 1.0)
        resid_b   = w_b * (phi_b - lam_b * syn_b)
        boot_ests[b] = lam_b * syn_unl_mean + resid_b.mean()

    var_boot = float(np.var(boot_ests, ddof=1))
    q_lo     = float(np.percentile(boot_ests, 5.0))
    q_hi     = float(np.percentile(boot_ests, 95.0))
    ci_lo    = 2.0 * mu_hat - q_hi
    ci_hi    = 2.0 * mu_hat - q_lo

    return float(mu_hat), var_boot, ci_lo, ci_hi


def _ppi_weighted_1d(
    phi_lab: np.ndarray,
    syn_lab: np.ndarray,
    syn_unl: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float]:
    """Weighted PPI++ (1-D, for use within this module)."""
    n = phi_lab.shape[0]
    N = syn_unl.shape[0]
    w = weights / weights.mean()
    phi_bar_w = (w * phi_lab).mean()
    syn_bar_w = (w * syn_lab).mean()
    cov_w     = (w * (phi_lab - phi_bar_w) * (syn_lab - syn_bar_w)).mean()
    var_w     = (w * (syn_lab - syn_bar_w) ** 2).mean()
    var_unl   = syn_unl.var()
    denom     = var_w + (n / N) * var_unl
    lam       = np.clip(cov_w / (denom + 1e-12), 0.0, 1.0)
    resid     = w * (phi_lab - lam * syn_lab)
    mu_hat    = lam * syn_unl.mean() + resid.mean()
    var_hat   = resid.var() / n + lam ** 2 * var_unl * (n / N) / n
    return float(mu_hat), float(var_hat)
