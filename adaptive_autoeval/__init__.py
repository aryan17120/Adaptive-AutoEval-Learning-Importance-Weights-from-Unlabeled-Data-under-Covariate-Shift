"""adaptive_autoeval -- Importance-weighted model evaluation under covariate shift.

This package implements Adaptive AutoEval: a statistically principled
framework for estimating model performance when labeled and unlabeled
data come from different distributions.

The key insight is to estimate importance weights w(x) = p_target(x) /
p_source(x) via discriminative classification, then integrate these
weights into the PPI++ variance framework.

Quick start::

    import numpy as np
    from adaptive_autoeval import ppi_weighted, learn_importance_weights

    # Your data
    feat_lab = ...   # (n, d) features of labeled (source) data
    feat_unl = ...   # (N, d) features of unlabeled (target) data
    phi_lab  = ...   # (n,) true loss values on labeled data
    syn_lab  = ...   # (n,) synthetic annotator predictions on labeled
    syn_unl  = ...   # (N,) synthetic annotator predictions on unlabeled

    # Step 1: Learn importance weights from data
    weights = learn_importance_weights(feat_lab, feat_unl)

    # Step 2: Adaptive AutoEval estimate + confidence interval
    mu_hat, var_hat = ppi_weighted(phi_lab, syn_lab, syn_unl, weights)

    # Step 3: Build 90% confidence interval
    from scipy.stats import norm
    z = norm.ppf(0.95)
    ci_lo = mu_hat - z * np.sqrt(var_hat)
    ci_hi = mu_hat + z * np.sqrt(var_hat)
    print(f"Estimate: {mu_hat:.3f}  90% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

For comparison, run the unweighted PPI++ baseline::

    from adaptive_autoeval import ppi_unweighted
    mu_base, var_base = ppi_unweighted(phi_lab, syn_lab, syn_unl)

See the paper for theoretical guarantees on consistency, asymptotic
normality, and effective sample size under unknown covariate shift.

Scope of the guarantee
----------------------
``ppi_weighted`` reduces bias and MSE relative to classical estimation
under unknown covariate shift.  It does **not** attain nominal coverage:
a residual coverage gap remains at every sample size tested.  See
``results/experiments.md`` for the measured gap on both benchmarks, and
``variance.sigma_weight_rate`` for why the weight-estimation variance
term is asymptotically negligible but does not close that gap.
"""

from .estimators import ppi_unweighted, ppi_weighted
from .weights import learn_importance_weights
from .variance import bootstrap_variance, sigma_weight_rate

__all__ = [
    "ppi_unweighted",
    "ppi_weighted",
    "learn_importance_weights",
    "bootstrap_variance",
    "sigma_weight_rate",
]

__version__ = "0.2.0"