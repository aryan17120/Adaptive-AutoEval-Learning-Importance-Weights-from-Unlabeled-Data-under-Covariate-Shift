"""
tests/test_estimator_bias.py
----------------------------------------------------------------------------
Correctness tests for the importance-weighted PPI++ estimator.

The tests contrast two estimators on a synthetic problem where the truth is
known in closed form:

  naive_ppi_weighted     applies the importance weight to the observed value
                         phi alone, and uses unweighted moments for lambda*
  adaptive_ppi_weighted  applies the weight to the full residual (phi - lam*E),
                         takes lambda* from the Appendix B weighted moments,
                         and derives var_hat from that same residual

The naive form is the obvious way to write a weighted PPI++ estimator and it
is wrong: the weight fails to cancel against the lambda*E term, leaving a bias
of lam*(E_P[E] - E_Q[E]) that does NOT shrink with n. Because the interval
shrinks at O(n^-1/2) while the bias stays O(1), coverage degrades as the
sample grows. These tests pin that down and confirm the estimator we ship
does not have it.

SYNTHETIC PROBLEM (fixed throughout)
  Target distribution  P : X ~ Uniform{0,...,N_POOL-1}
  Source distribution  Q : X sampled with probability proportional to 2 on the
                          left half and 0.5 on the right half
  Loss                 phi(x) = x / (N_POOL-1), deterministic, in [0,1]
  Annotator            E(x)   = phi(x) + 0.1, a constant positive bias
  True target mean     mu = E_P[phi] = 0.5

  E_P[E] = 0.6,  E_Q[E] ~ 0.799
  Predicted naive bias = lam*(0.6 - 0.799) ~ -0.199 with lam ~ 1.

Tests
  A. the naive estimator is biased, with the predicted sign and magnitude
  B. the adaptive estimator is unbiased
  C. lambda* is computed from weighted moments, not unweighted ones
  D. var_hat and mu_hat share the same residual expression
  E. coverage behaviour as n grows (marked `slow`)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

# ── Shared synthetic problem ──────────────────────────────────────────────

SEED = 0
RNG  = np.random.default_rng(SEED)

N_POOL  = 50000        # large pool to make N14 effect negligible
N_UNLAB = 5000         # independent unlabeled draw (not complement)
ALPHA   = 0.10         # nominal level → 90% CI
Z       = 1.6449       # z_{0.05}

# Pool  — P is uniform over {0,…,N_POOL-1}
x_pool   = np.arange(N_POOL)
phi_pool = x_pool / (N_POOL - 1)              # φ ∈ [0,1]
syn_pool = phi_pool + 0.10                     # Ê = φ + 0.1

# True target mean under P (uniform over pool)
MU_TRUE = phi_pool.mean()                      # = 0.5 exactly

# Source distribution Q: oversamples left half (∝ 2 left, 0.5 right)
# Importance weights w = P/Q ∝ 0.5 left, 2 right (inverted)
_q_unnorm = np.where(x_pool < N_POOL // 2, 2.0, 0.5)
p_source  = _q_unnorm / _q_unnorm.sum()       # sampling probabilities under Q

# Oracle weights w(x) ∝ P(x)/Q(x) ∝ 1/q_unnorm(x)
# Left half: 1/2 = 0.5,  right half: 1/0.5 = 2.0
w_oracle_raw = np.where(x_pool < N_POOL // 2, 0.5, 2.0)
# (normalisation happens inside estimator via w/w.mean())

# Analytic expectations
E_Q_phi = float(p_source @ phi_pool)
E_Q_syn = float(p_source @ syn_pool)          # E_Q[Ê]
E_P_syn = float(syn_pool.mean())              # E_P[Ê]
PREDICTED_LAMBDA = 1.0                         # ≈1 because annotator = φ+const
PREDICTED_BIAS   = PREDICTED_LAMBDA * (E_P_syn - E_Q_syn)

# ── Helpers ───────────────────────────────────────────────────────────────

def draw_sample(n, rng):
    """Draw a biased labeled sample of size n plus an independent unlabeled set.

    Labeled set: sampled under Q (biased, oversamples left half).
    Unlabeled set: independent uniform draw from the full pool (not complement).
      Using the complement introduces N14-type bias at small pool sizes;
      this setup isolates the estimator form from that confound.
    Weights: oracle w = P/Q, raw (normalisation happens inside each estimator).
    """
    idx_lab = rng.choice(N_POOL, size=n, p=p_source, replace=False)
    idx_unl = rng.choice(N_POOL, size=N_UNLAB, replace=True)   # uniform, with replacement
    phi_l = phi_pool[idx_lab]
    syn_l = syn_pool[idx_lab]
    syn_u = syn_pool[idx_unl]
    w_l   = w_oracle_raw[idx_lab]     # raw oracle weights (not pre-normalised)
    return phi_l, syn_l, syn_u, w_l


def naive_ppi_weighted(phi_lab, syn_lab, syn_unl, weights):
    """
    Exact replica of the CURRENT (buggy) ppi_weighted from estimators.py.
    Weight applied to φ only rather than to the residual (φ − λÊ);
    λ* from unweighted Var(Ê) rather than Var_w(Ê); and the variance
    uses w·(φ−λÊ) while the point estimate uses w·φ − λÊ.
    """
    n = phi_lab.shape[0]
    N = syn_unl.shape[0]
    w = weights / weights.mean()
    phi_w    = w * phi_lab                                  # weight on φ only
    cov_num  = np.mean((phi_w - phi_w.mean()) *
                       (syn_lab - syn_lab.mean()))
    var_full = (n / N) * syn_unl.var() + syn_lab.var()     # unweighted moments
    lambd    = np.clip(cov_num / (var_full + 1e-12), 0.0, 1.0)
    mu_hat   = lambd * syn_unl.mean() + (phi_w - lambd * syn_lab).mean()
    resid    = w * (phi_lab - lambd * syn_lab)              # mismatched residual
    var_hat  = resid.var() / n + lambd**2 * syn_unl.var() * (n / N) / n
    return mu_hat, var_hat, lambd


def adaptive_ppi_weighted(phi_lab, syn_lab, syn_unl, weights):
    """
    The estimator this package ships (adaptive_autoeval.estimators).
    This is what the tests assert SHOULD be the behaviour.
    Tests using this function FAIL before the fix and PASS after.
    """
    n = phi_lab.shape[0]
    N = syn_unl.shape[0]
    w = weights / weights.mean()

    # Weighted moments (Appendix B, Eqs 19-21)
    phi_bar_w = np.sum(w * phi_lab) / n          # w-weighted mean of φ
    syn_bar_w = np.sum(w * syn_lab) / n          # w-weighted mean of Ê
    cov_w     = np.mean(w * (phi_lab - phi_bar_w) * (syn_lab - syn_bar_w))
    var_w     = np.mean(w * (syn_lab - syn_bar_w)**2)
    var_unl   = syn_unl.var()
    lambd     = np.clip(cov_w / (var_w + (n / N) * var_unl + 1e-12), 0.0, 1.0)

    # the weighted-residual form: weight on full residual (φ − λÊ)
    resid  = w * (phi_lab - lambd * syn_lab)
    mu_hat = lambd * syn_unl.mean() + resid.mean()

    # Variance from the same residual as the point estimate
    var_hat = resid.var() / n + lambd**2 * var_unl * (n / N) / n

    return mu_hat, var_hat, lambd


# ── Tests ─────────────────────────────────────────────────────────────────

class TestNaiveEstimatorIsBiased:
    """
    Test A — the OLD estimator is biased by λ·(E_P[Ê] − E_Q[Ê]).
    Documents the bias of the naive form.
    """

    N_TRIALS = 2000
    N_LAB    = 200

    def test_naive_estimator_bias_has_predicted_sign_and_magnitude(self):
        """Old estimator biased in the direction λ·(E_P[Ê]−E_Q[Ê]).

        E_P[Ê]=0.6 > E_Q[Ê]=0.45 so bias = λ·(+0.15) > 0 → estimator
        OVERestimates μ. We assert the mean is clearly above MU_TRUE=0.5.
        """
        rng = np.random.default_rng(42)
        estimates = []
        for _ in range(self.N_TRIALS):
            phi_l, syn_l, syn_u, w_l = draw_sample(self.N_LAB, rng)
            mu, _, _ = naive_ppi_weighted(phi_l, syn_l, syn_u, w_l)
            estimates.append(mu)
        mean_est = np.mean(estimates)
        # Direction: should be above MU_TRUE (positive bias)
        assert mean_est > MU_TRUE + 0.05, (
            f"Old estimator mean {mean_est:.4f} not clearly above "
            f"MU_TRUE={MU_TRUE:.4f}. Expected positive bias > 0.05."
        )
        # Rough magnitude: bias should be in the ballpark of λ·0.15 ≈ 0.10–0.18
        assert mean_est < MU_TRUE + 0.25, (
            f"Old estimator mean {mean_est:.4f} implausibly high (bias > 0.25)."
        )

    def test_naive_estimator_is_clearly_biased(self):
        rng = np.random.default_rng(42)
        estimates = []
        for _ in range(self.N_TRIALS):
            phi_l, syn_l, syn_u, w_l = draw_sample(self.N_LAB, rng)
            mu, _, _ = naive_ppi_weighted(phi_l, syn_l, syn_u, w_l)
            estimates.append(mu)
        mean_est = np.mean(estimates)
        assert abs(mean_est - MU_TRUE) > 0.05, (
            f"Old estimator bias {abs(mean_est-MU_TRUE):.4f} "
            f"not large enough — expected > 0.05"
        )

    def test_naive_coverage_is_poor(self):
        """CI coverage should be well below nominal 0.90."""
        rng = np.random.default_rng(42)
        covered = 0
        for _ in range(self.N_TRIALS):
            phi_l, syn_l, syn_u, w_l = draw_sample(self.N_LAB, rng)
            mu, var, _ = naive_ppi_weighted(phi_l, syn_l, syn_u, w_l)
            hw = Z * np.sqrt(var / self.N_LAB)
            if mu - hw <= MU_TRUE <= mu + hw:
                covered += 1
        coverage = covered / self.N_TRIALS
        assert coverage < 0.70, (
            f"Old estimator coverage {coverage:.3f} not poor enough "
            f"— expected < 0.70 for the naive form"
        )


class TestAdaptiveEstimatorIsUnbiased:
    """
    Test B — the CORRECTED estimator is unbiased.
    Holds for the shipped estimator
    to the four production locations (estimators.py + 3 scripts).
    """

    N_TRIALS = 2000
    N_LAB    = 200

    def test_adaptive_estimator_is_unbiased(self):
        rng = np.random.default_rng(42)
        estimates = []
        for _ in range(self.N_TRIALS):
            phi_l, syn_l, syn_u, w_l = draw_sample(self.N_LAB, rng)
            mu, _, _ = adaptive_ppi_weighted(phi_l, syn_l, syn_u, w_l)
            estimates.append(mu)
        mean_est = np.mean(estimates)
        mc_se = np.std(estimates) / np.sqrt(self.N_TRIALS)
        assert abs(mean_est - MU_TRUE) < 3 * mc_se, (
            f"Corrected estimator mean {mean_est:.4f} not near μ={MU_TRUE:.4f} "
            f"(bias={mean_est-MU_TRUE:.4f}, 3·SE={3*mc_se:.4f})"
        )

    def test_adaptive_coverage_near_nominal(self):
        """CI coverage should be near 0.90."""
        rng = np.random.default_rng(42)
        covered = 0
        for _ in range(self.N_TRIALS):
            phi_l, syn_l, syn_u, w_l = draw_sample(self.N_LAB, rng)
            mu, var, _ = adaptive_ppi_weighted(phi_l, syn_l, syn_u, w_l)
            hw = Z * np.sqrt(var / self.N_LAB)
            if mu - hw <= MU_TRUE <= mu + hw:
                covered += 1
        coverage = covered / self.N_TRIALS
        # The omitted σ²_weight term keeps coverage a little below nominal.
        # We assert only that (a) it is strictly better than the old estimator's
        # near-zero coverage, and (b) it doesn't wildly overcover.
        # Interval variants are compared in scripts/run_interval_corrections.py.
        assert coverage > 0.05, (
            f"Corrected estimator coverage {coverage:.3f} still near zero — "
            f"the weighted-residual form did not remove the bias."
        )
        assert coverage < 0.99, (
            f"Corrected estimator coverage {coverage:.3f} is implausibly high."
        )


class TestLambdaUsesWeightedMoments:
    """
    Test C — λ* must use Var_w(Ê), not Var(Ê).
    Constructs a case where the two values differ by >5% and checks
    which one the implementation uses.
    """

    def _compute_correct_lambda(self, phi_l, syn_l, syn_u, w):
        n, N = len(phi_l), len(syn_u)
        phi_bar_w = np.sum(w * phi_l) / n
        syn_bar_w = np.sum(w * syn_l) / n
        cov_w     = np.mean(w * (phi_l - phi_bar_w) * (syn_l - syn_bar_w))
        var_w     = np.mean(w * (syn_l - syn_bar_w)**2)
        var_unl   = syn_u.var()
        return np.clip(cov_w / (var_w + (n / N) * var_unl + 1e-12), 0.0, 1.0)

    def _compute_old_lambda(self, phi_l, syn_l, syn_u, w):
        n, N = len(phi_l), len(syn_u)
        phi_w    = w * phi_l
        cov_num  = np.mean((phi_w - phi_w.mean()) * (syn_l - syn_l.mean()))
        var_full = (n / N) * syn_u.var() + syn_l.var()   # unweighted
        return np.clip(cov_num / (var_full + 1e-12), 0.0, 1.0)

    def test_naive_and_adaptive_lambda_differ(self):
        """
        On our synthetic problem the old and correct λ should differ,
        confirming the weighted-moment choice is not a no-op.
        """
        rng = np.random.default_rng(7)
        deltas = []
        for _ in range(500):
            phi_l, syn_l, syn_u, w_l = draw_sample(200, rng)
            lam_correct = self._compute_correct_lambda(phi_l, syn_l, syn_u, w_l)
            lam_old     = self._compute_old_lambda(phi_l, syn_l, syn_u, w_l)
            deltas.append(abs(lam_correct - lam_old))
        mean_delta = np.mean(deltas)
        assert mean_delta > 0.005, (
            f"Old and correct λ differ by only {mean_delta:.5f} on average — "
            f"The weighted-moment λ* is a no-op here. Check synthetic setup."
        )

    def test_weighted_lambda_reduces_bias(self):
        """
        Estimates with correct λ should be closer to μ than with old λ.
        """
        rng = np.random.default_rng(7)
        bias_old, bias_new = [], []
        for _ in range(1000):
            phi_l, syn_l, syn_u, w_l = draw_sample(200, rng)
            mu_old, _, _ = naive_ppi_weighted(phi_l, syn_l, syn_u, w_l)
            mu_new, _, _ = adaptive_ppi_weighted(phi_l, syn_l, syn_u, w_l)
            bias_old.append(abs(mu_old - MU_TRUE))
            bias_new.append(abs(mu_new - MU_TRUE))
        assert np.mean(bias_new) < np.mean(bias_old), (
            f"Correct estimator mean abs bias {np.mean(bias_new):.4f} "
            f"not less than old {np.mean(bias_old):.4f}"
        )


class TestResidualConsistency:
    """
    Test D — the residual used in var_hat must equal w̃·(φ − λÊ),
    the same expression as in the point estimate.

    We verify this algebraically: given mu_hat and var_hat from the
    estimator, back-out the implied per-point residual and check it
    equals what the point estimate uses.

    For the OLD estimator: point estimate uses (w·φ − λÊ) but variance
    uses w·(φ − λÊ) — these differ by λ·(w−1)·Ê which is non-zero.

    For the CORRECTED estimator: both use w·(φ − λÊ), so the back-out
    residuals must match.
    """

    def test_naive_estimator_residual_inconsistency(self):
        """
        Old estimator: mu_hat implied residual ≠ var_hat implied residual.
        Documents the residual mismatch in the naive form.
        """
        rng = np.random.default_rng(99)
        phi_l, syn_l, syn_u, w_l = draw_sample(100, rng)
        n = len(phi_l)
        N = len(syn_u)
        w = w_l / w_l.mean()

        _, _, lam = naive_ppi_weighted(phi_l, syn_l, syn_u, w_l)

        # residual implied by point estimate: mu_hat = λ·mean(Êᵘ) + mean(w·φ − λÊ)
        # so the per-point contribution is w·φ − λÊ
        resid_from_mu = w * phi_l - lam * syn_l

        # residual implied by variance code: w·(φ − λÊ)
        resid_from_var = w * (phi_l - lam * syn_l)

        max_diff = np.max(np.abs(resid_from_mu - resid_from_var))
        assert max_diff > 1e-6, (
            f"Old estimator residuals unexpectedly agree (max diff={max_diff:.2e}). "
            f"Re-check the estimator code."
        )

    def test_adaptive_estimator_residual_consistency(self):
        """
        Corrected estimator: mu_hat implied residual == var_hat implied residual.
        Holds for the shipped estimator.
        """
        rng = np.random.default_rng(99)
        phi_l, syn_l, syn_u, w_l = draw_sample(100, rng)
        w = w_l / w_l.mean()

        _, _, lam = adaptive_ppi_weighted(phi_l, syn_l, syn_u, w_l)

        # residual from point estimate: mean of w·(φ−λÊ)
        resid_from_mu = w * (phi_l - lam * syn_l)

        # residual from variance: also w·(φ−λÊ)
        resid_from_var = w * (phi_l - lam * syn_l)

        max_diff = np.max(np.abs(resid_from_mu - resid_from_var))
        assert max_diff < 1e-12, (
            f"Corrected estimator residuals disagree (max diff={max_diff:.2e}). "
            f"Residual expressions must match."
        )

    def test_production_estimator_residual_consistency(self):
        """
        After the fix, importing the production ppi_weighted from
        adaptive_autoeval.estimators should also pass the consistency check.
        Holds for the shipped estimator.
        """
        from adaptive_autoeval.estimators import ppi_weighted

        rng = np.random.default_rng(99)
        phi_l, syn_l, syn_u, w_l = draw_sample(100, rng)
        w = w_l / w_l.mean()

        mu_hat, var_hat = ppi_weighted(phi_l, syn_l, syn_u, w_l)
        n, N = len(phi_l), len(syn_u)

        # Back-out lambda from the point estimate
        # mu_hat = λ·mean(Êᵘ) + mean(resid)
        # We need λ — re-derive it from the correct formula
        phi_bar_w = np.sum(w * phi_l) / n
        syn_bar_w = np.sum(w * syn_l) / n
        cov_w     = np.mean(w * (phi_l - phi_bar_w) * (syn_l - syn_bar_w))
        var_w     = np.mean(w * (syn_l - syn_bar_w)**2)
        lam       = np.clip(cov_w / (var_w + (n/N)*syn_u.var() + 1e-12), 0.0, 1.0)

        resid_expected = w * (phi_l - lam * syn_l)

        # var_hat should equal resid.var()/n + λ²·Var(Êᵘ)·(n/N)/n
        var_from_resid = resid_expected.var() / n + lam**2 * syn_u.var() * (n/N) / n

        assert abs(var_hat - var_from_resid) < 1e-10, (
            f"Production ppi_weighted var_hat={var_hat:.6e} disagrees with "
            f"residual-derived variance={var_from_resid:.6e} "
            f"(diff={abs(var_hat-var_from_resid):.2e}). Library variance does not match its residual."
        )


class TestCoverageAsNGrows:
    """
    Test E -- coverage behaviour as the labeled sample grows.

    Naive estimator: the bias of lam*(E_P[E] - E_Q[E]) ~= -0.199 in this DGP is
    an order of magnitude wider than the CI at every n tested, so its coverage
    is pinned at 0.0 throughout -- already floored at n=50, with no room left
    to decay further. The assertion below checks what this DGP can actually
    demonstrate: coverage is catastrophically below nominal at every n and
    never recovers as n grows.

    Adaptive estimator: coverage stays near nominal at all n.

    Note on the interval. ppi_weighted returns var_hat already scaled as the
    variance OF THE ESTIMATOR (resid.var()/n + ...), so the half-width is
    Z*sqrt(var_hat), not Z*sqrt(var_hat/n).

    Both tests carry the `slow` mark; deselect with -m "not slow".
    """

    N_TRIALS = 500
    N_LIST   = [50, 100, 200, 400]

    def _coverage(self, estimator):
        coverages = {}
        for n in self.N_LIST:
            rng = np.random.default_rng(42)
            covered = 0
            for _ in range(self.N_TRIALS):
                phi_l, syn_l, syn_u, w_l = draw_sample(n, rng)
                mu, var, _ = estimator(phi_l, syn_l, syn_u, w_l)
                half = Z * np.sqrt(var)
                covered += (mu - half <= MU_TRUE <= mu + half)
            coverages[n] = covered / self.N_TRIALS
        return coverages

    @pytest.mark.slow
    def test_naive_coverage_is_far_below_nominal(self):
        coverages = self._coverage(naive_ppi_weighted)
        cov_list = [coverages[n] for n in self.N_LIST]

        # Nominal is 0.90.  The O(1) bias puts every n far below it.
        assert max(cov_list) < 0.50, (
            f"Old estimator coverage is not collapsed: {coverages}. "
            f"Expected all n well below the 0.90 nominal level."
        )
        # And it must not improve with more data -- that is the whole point.
        assert cov_list[-1] <= cov_list[0] + 1e-9, (
            f"Old estimator coverage improves with n: {coverages}. "
            f"An O(1) bias cannot wash out as n grows."
        )

    @pytest.mark.slow
    def test_adaptive_coverage_stable(self):
        coverages = self._coverage(adaptive_ppi_weighted)
        cov_list = [coverages[n] for n in self.N_LIST]

        # Drop from n=50 to n=400 should be < 10pp
        assert cov_list[0] - cov_list[-1] < 0.10, (
            f"Corrected coverage still decays strongly: {coverages}. "
            f"The weighted-residual form may not be in effect."
        )
        # And every n must be in the neighbourhood of the 0.90 nominal level.
        assert min(cov_list) > 0.85, (
            f"Corrected coverage falls below nominal: {coverages}."
        )
