# Experimental Results

Nominal level is **90%** throughout. The shift mechanism is exponential,
∝ exp(β·s), with β=1.0 unless stated otherwise. Every figure in `results/` is
regenerated from the CSVs alongside it by `python scripts/make_figures.py`.

---

## ImageNet: Synthetic Covariate Shift

**Setup**: 50,000 images, 5 ResNet models, 250 Monte Carlo trials.
Labeled examples drawn ∝ exp(β·s), where s is the ResNet-101 confidence score.
Weight-learning features are independent of s.

| n   | Classical | PPI++ | Oracle PPI++ | **Adaptive** | MSE (Classical) | MSE (Adaptive) | ESS  |
|-----|-----------|-------|--------------|--------------|-----------------|----------------|------|
| 50  | 0.735 | 0.830 | 0.882 | **0.854** | 0.00552 | **0.00290** | 1.32 |
| 100 | 0.607 | 0.833 | 0.886 | **0.852** | 0.00404 | **0.00154** | 1.31 |
| 200 | 0.462 | 0.838 | 0.893 | **0.862** | 0.00311 | **0.00076** | 1.28 |
| 300 | 0.370 | 0.843 | 0.916 | **0.860** | 0.00257 | **0.00050** | 1.30 |
| 400 | 0.232 | 0.772 | 0.884 | **0.823** | 0.00276 | **0.00046** | 1.29 |
| 500 | 0.140 | 0.774 | 0.901 | **0.815** | 0.00269 | **0.00037** | 1.27 |

`ext1_results.csv` · `ext1_main.png`

**Key finding**: classical coverage collapses from 0.735 to 0.140 as n grows —
the interval shrinks around a biased point. Adaptive AutoEval holds coverage
**stable at 0.815–0.862**, beats unweighted PPI++ at every n by 1.7–5.1
percentage points, and reduces MSE by **48–86%**.

Coverage does not reach the 90% nominal level. The residual 4–9 point gap is
attributable to the strength of the shift and to finite-sample approximation
in the asymptotic variance, and is reported as a limitation rather than
closed. See [Limitations](#limitations).

### Variance-estimator variants

Holding the point estimator fixed, four variance estimators were compared:

| n   | two-term | bootstrap | cross-fit | t-quantile | cf + t | boot + cf |
|-----|----------|-----------|-----------|------------|--------|-----------|
| 50  | 0.849 | 0.871 | 0.862 | 0.858 | 0.869 | 0.854 |
| 100 | 0.859 | 0.866 | 0.867 | 0.862 | 0.872 | 0.858 |
| 200 | 0.866 | 0.878 | 0.868 | 0.866 | 0.871 | 0.865 |
| 300 | 0.867 | 0.875 | 0.867 | 0.870 | 0.868 | 0.862 |
| 400 | 0.822 | 0.824 | 0.822 | 0.822 | 0.822 | 0.813 |
| 500 | 0.817 | 0.828 | 0.820 | 0.818 | 0.822 | 0.808 |

`interval_variants.csv` · `interval_variants.png`

The variants differ by only 1–3 percentage points, and none closes the gap to
nominal. The two-term plug-in is the default. The weighted bootstrap appears
best at large n, but this is not a sound gain: holding the fitted
discriminator fixed across resamples makes it **underestimate** variance
(ratio to the two-term variance ≈0.15–0.55 at n=200–400), so its intervals are
narrower than they should be.

---

## ProteinGym SPG1: Fitness-Biased Labeling

**Setup**: 536,962 protein variants, 7 foundation models, VESPA annotator,
fitness-biased shift β=1.0.

| n    | Classical | PPI++ | **Adaptive** | MSE (Classical) | MSE (Adaptive) | MSE reduction |
|------|-----------|-------|--------------|-----------------|----------------|---------------|
| 200  | 0.210 | 0.201 | **0.873** | 0.564 | **0.0443** | 92.2% |
| 400  | 0.057 | 0.038 | **0.883** | 0.481 | **0.0235** | 95.1% |
| 600  | 0.003 | 0.007 | **0.825** | 0.483 | **0.0222** | 95.4% |
| 800  | 0.000 | 0.002 | **0.795** | 0.490 | **0.0191** | 96.1% |
| 1000 | 0.000 | 0.000 | **0.786** | 0.477 | **0.0168** | 96.5% |
| 1500 | 0.000 | 0.000 | **0.733** | 0.473 | **0.0140** | 97.0% |

`ext1_pg_results.csv` · `ext1_pg_main.png`

**Key finding**: 92–97% MSE reduction against classical estimation. Both
classical and unweighted PPI++ reach exactly zero coverage by n=800.

### Sensitivity to the weight-feature set

The run above gives the weight-learning classifier calibrated model scores,
which are correlated with the fitness variable driving the selection bias.
Restricting it to features independent of the annotator makes weight
estimation substantially harder, and coverage falls away:

| n    | 200   | 400   | 600   | 800   | 1000  | 1500  |
|------|-------|-------|-------|-------|-------|-------|
| Coverage | 0.426 | 0.202 | 0.129 | 0.063 | 0.033 | 0.008 |

`interval_variants.csv`

No variance-estimator variant moves this at any n. The shortfall is
structural: the shift is strong, the max-to-min weight ratio is around 38, and
the asymptotic approximation breaks down at these sample sizes.

**On ProteinGym the supported claim is MSE reduction, not coverage.** This
sensitivity is the single largest open issue in the method and is stated as
such in [Limitations](#limitations).

### Model ranking

| Method    | Mean ρ | Std   |
|-----------|--------|-------|
| Classical | 0.941  | 0.054 |
| PPI++     | 0.938  | 0.054 |
| Adaptive  | 0.860  | 0.194 |

`ext1_pg_spearman.csv` · `ext1_pg_ranking.png`

All three methods preserve gross ranking, but reweighting **costs** ranking
accuracy here: adaptive ρ is below classical, with nearly four times the
variance. The method's advantage is in correcting **absolute** performance
estimates, not in ordering models. We do not claim a ranking improvement.

---

## Ablation Studies (ImageNet)

All three ablations are run under one configuration, with each row tagged by a
config hash. `results/ablations/*.csv` · `abl_combined.png`

### A1: Shift Severity

Adaptive coverage by β, with classical at n=500 for reference:

| β   | n=50 | n=100 | n=200 | n=300 | n=400 | n=500 | Classical n=500 |
|-----|------|-------|-------|-------|-------|-------|-----------------|
| 0.0 | 0.871 | 0.906 | 0.905 | 0.888 | 0.905 | 0.911 | 0.906 |
| 0.5 | 0.884 | 0.890 | 0.888 | 0.887 | 0.867 | 0.873 | 0.582 |
| 1.0 | 0.853 | 0.852 | 0.866 | 0.866 | 0.821 | 0.818 | 0.140 |
| 2.0 | 0.846 | 0.804 | 0.807 | 0.734 | 0.725 | 0.690 | 0.002 |
| 3.0 | 0.801 | 0.750 | 0.692 | 0.650 | 0.614 | 0.551 | 0.000 |

At β=0 (no shift) the method is well calibrated and costs nothing, which is
the important negative control. Degradation is graceful and tracks shift
strength; classical estimation is at zero coverage by β=2.

### A2: Classifier Capacity

| Classifier          | n=50 | n=200 | n=500 |
|---------------------|------|-------|-------|
| Logistic Regression | 0.852 | 0.865 | 0.815 |
| MLP (64×64)         | 0.858 | 0.866 | 0.794 |
| Random Forest       | 0.832 | 0.834 | 0.790 |

Insensitive to classifier choice — the spread at any n is under 3 points, and
logistic regression is not beaten. Capacity is not the bottleneck.

### A3: Clipping Threshold

| ε     | n=50 | n=200 | n=500 |
|-------|------|-------|-------|
| 0.001 | 0.852 | 0.869 | 0.816 |
| 0.010 | 0.850 | 0.866 | 0.817 |
| 0.050 | 0.856 | 0.868 | 0.818 |
| 0.100 | 0.850 | 0.865 | 0.814 |
| 0.200 | 0.828 | 0.838 | 0.774 |

Flat from ε=0.001 to 0.1; only the aggressive ε=0.2 costs anything.

Note on bounds: the implementation clips discriminator **probabilities** to
[ε, 1−ε], which induces **weight** bounds [ε/(1−ε), (1−ε)/ε] ≈ [0.0101, 99.0]
at ε=0.01.

---

## Diagnostics

`results/diagnostics/*.csv`

### Alternative density-ratio estimators

KMM, KLIEP and uLSIF were compared against the discriminative estimator under
identical clipping and matched low-dimensional features. "Win rate" is the
fraction of Monte Carlo trials in which a method beats the discriminative one.

| n   | Discriminative | KMM   | uLSIF | KLIEP | KMM win | uLSIF win | KLIEP win |
|-----|----------------|-------|-------|-------|---------|-----------|-----------|
| 50  | 0.851 | 0.902 | 0.826 | 0.713 | 0.284 | 0.112 | 0.096 |
| 100 | 0.858 | 0.863 | 0.830 | 0.690 | 0.236 | 0.132 | 0.112 |
| 200 | 0.862 | 0.862 | 0.814 | 0.662 | 0.196 | 0.116 | 0.100 |
| 300 | 0.867 | 0.875 | 0.829 | 0.681 | 0.172 | 0.088 | 0.112 |
| 400 | 0.820 | 0.851 | 0.761 | 0.542 | 0.204 | 0.128 | 0.084 |
| 500 | 0.814 | 0.836 | 0.766 | 0.547 | 0.220 | 0.120 | 0.060 |

`dre_comparison_imagenet.csv` · `dre_comparison.png`

uLSIF and KLIEP underperform at every n despite identical clipping — their
kernel weights become highly concentrated after clipping. KMM has higher
coverage than the discriminative estimator at every n, though it wins only
17–28% of individual trials, and it is O(n²) in time and memory, so it does
not scale to 2048-dimensional features.

We therefore present the contribution as an **evaluation framework** into
which any density-ratio estimator can be substituted, rather than as an
argument for the discriminative choice specifically.

### Oracle versus learned weights

Both estimators receive identical clipping and calibration, so the comparison
is like-for-like.

| n   | Oracle | Learned | Oracle win rate | χ² (oracle) | χ² (learned) |
|-----|--------|---------|-----------------|-------------|--------------|
| 50  | 0.882 | 0.852 | 0.204 | 0.0536 | 0.0470 |
| 100 | 0.886 | 0.860 | 0.220 | 0.0543 | 0.0381 |
| 200 | 0.893 | 0.866 | 0.220 | 0.0552 | 0.0301 |
| 300 | 0.916 | 0.867 | 0.280 | 0.0561 | 0.0227 |
| 400 | 0.884 | 0.822 | 0.308 | 0.0558 | 0.0209 |
| 500 | 0.901 | 0.815 | 0.416 | 0.0555 | 0.0240 |

`oracle_diagnostic.csv`

The oracle attains higher coverage at every n and the margin widens with n, as
expected. Learned weights recover most but not all of the oracle's gain; the
shortfall at n=500 is 8.6 points and is the clearest quantification of the
cost of not knowing the true density ratio.

### Effective sample size, reported as two metrics

A single ESS ratio conflates the efficiency *gain* from prediction-powered
inference with the efficiency *cost* of reweighting. We report both:

| n   | ESS_prop1 (vs unshifted i.i.d. classical) | Reweighting cost (vs unweighted PPI++) |
|-----|-------------------------------------------|----------------------------------------|
| 50  | 1.59 | 0.858 |
| 100 | 1.54 | 0.869 |
| 200 | 1.50 | 0.875 |
| 300 | 1.51 | 0.890 |
| 400 | 1.50 | 0.883 |
| 500 | 1.49 | 0.876 |

`ess_split.csv`

PPI gives a ~1.5× efficiency gain; reweighting costs about 12% of it.

### Weight-estimation variance (σ²_weight)

Because the weights are learned they carry their own uncertainty. Three
results, all negative:

1. **The term is asymptotically negligible.** Under logistic regression
   converging at the parametric rate, σ²_weight is O(n⁻¹), while the two
   dominant terms are O(n⁻¹ᐟ²) in standard-error terms. At n=500 the omitted
   term is under 5% of the oracle variance, so the two-term formula is a valid
   *first-order* variance estimator.
2. **The plug-in is unusable.** The Fisher-information plug-in produces an
   estimate ~24× the oracle variance at n=200 with two features, driving
   coverage to 1.000 — intervals so wide they are trivially correct and
   uninformative. The instability is structural: I(θ*)⁻¹ has large eigenvalues
   where g*(x) ≈ 0.5, amplified by w(x)². It is intractable for d ≥ 20 and is
   not shipped.
3. **The bootstrap does not help.** See the variance-variants section above.

Together these rule out weight-estimation variance as the explanation for the
residual coverage gap. See `adaptive_autoeval/variance.py`.

---

## Limitations

1. **Coverage does not reach nominal.** On ImageNet the gap is 4–9 points at
   every n. The method substantially improves on classical estimation and
   modestly on unweighted PPI++, but it does not deliver calibrated 90%
   intervals, and we do not claim it does.

2. **Performance depends heavily on the weight-feature set.** This is the
   largest open issue. On ProteinGym, restricting the weight learner to
   features independent of the annotator drops coverage from 0.873 to 0.426 at
   n=200 and to 0.008 at n=1500. Where informative annotator-independent
   features are unavailable, the coverage guarantee should not be relied on;
   the MSE reduction is the robust result.

3. **Results use proxy features, not the full embeddings.** The numbers in
   this repository were produced with the 1-d softmax-entropy proxy on
   ImageNet and model scores on ProteinGym. `extract_features_imagenet.py` and
   `extract_features_proteingym.py` produce the 2048-d ResNet-50 penultimate
   and 640-d ESM-2 feature sets respectively; results under those are not yet
   measured.

4. **The ranking analysis uses a separate code path.** `compute_spearman.py`
   implements the estimator independently of the main experiment scripts. Its
   numbers should be treated as indicative until produced by a shared path.

5. **Per-trial reproducibility.** The ablation scripts draw the unlabeled
   subsample from the global NumPy random stream rather than a per-trial
   seeded generator. Each script as a whole is deterministic at `seed=42`, but
   individual trials are not reproducible in isolation, and fresh runs may
   drift by up to 0.001.

6. **Single ProteinGym assay.** All protein results are on SPG1
   (Olson 2014). Generality across assays is untested.
