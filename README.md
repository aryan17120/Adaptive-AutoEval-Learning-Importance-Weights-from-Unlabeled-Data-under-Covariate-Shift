# Adaptive AutoEval: Learning Importance Weights from Unlabeled Data under Covariate Shift
### Importance-Weighted Model Evaluation under Unknown Covariate Shift
*Anonymous submission — under review*

---

## The Problem

Modern model evaluation pipelines rely on the AutoEval framework,
which combines small human-labeled datasets with large synthetic-label
datasets via prediction-powered inference (PPI++) to produce
statistically valid performance estimates. It works well — but only
when labeled and unlabeled data come from the same distribution.

In practice, this assumption is routinely violated. Consider an
ImageNet evaluation where annotators label high-confidence, easy
images first. A model reads the accuracy correctly on this biased
labeled set — but the estimate reflects the easy subset, not the
true target distribution. Standard PPI++ marks this as a valid
confidence interval. The shift is invisible to existing metrics.

This failure mode — which we call **covariate shift misevaluation**
— is pervasive whenever labeled data is collected under any selection
bias: confidence-based sampling, cost-driven annotation, or
assay-specific experimental conditions. No existing AutoEval method
corrects for it without knowing the true density ratio.

Adaptive AutoEval fills this gap with:

- A formal **importance-weighted PPI++ estimator** that corrects for
  unknown covariate shift by learning density ratios from data
- **Theoretical guarantees** on consistency, asymptotic normality,
  and effective sample size under weight estimation error
- **Empirical measurements** of how much of the misevaluation this
  removes across two domains, and how much remains

---

## Key Results

We tested Adaptive AutoEval on ImageNet (vision) and ProteinGym
(protein biology) under realistic labeling biases. Nominal level is
90% throughout; the shift is exponential, ∝ exp(β·s), with β=1.0.

**Headline finding**: classical estimation collapses under moderate
selection bias — coverage falls to **14%** at n=500 on ImageNet and to
**0%** on ProteinGym — while importance weighting holds coverage
roughly stable and cuts error by up to two orders of magnitude.

### ImageNet — Synthetic Covariate Shift

| n   | Classical | PPI++ | Oracle | **Adaptive** | Pattern |
|-----|-----------|-------|--------|--------------|---------|
| 50  | 0.735 | 0.830 | 0.882 | **0.854** | All weighted methods near nominal |
| 100 | 0.607 | 0.833 | 0.886 | **0.852** | Classical already degrading |
| 200 | 0.462 | 0.838 | 0.893 | **0.862** | Adaptive best non-oracle |
| 300 | 0.370 | 0.843 | 0.916 | **0.860** | Stable in n |
| 400 | 0.232 | 0.772 | 0.884 | **0.823** | Classical collapsing |
| 500 | 0.140 | 0.774 | 0.901 | **0.815** | Adaptive 5.8× classical |

Coverage of nominal 90% confidence intervals (250 trials).
Adaptive AutoEval is **stable at 0.815–0.862** across the range,
beats unweighted PPI++ at every n by 1.7–5.1 points, and reduces MSE
by **48–86%** relative to classical estimation. It does not reach the
90% nominal level; the residual 4–9 point gap is reported as a
limitation, not closed.

### ProteinGym SPG1 — Fitness-Biased Labeling (β=1.0)

| n    | Classical | PPI++ | **Adaptive** | MSE (Classical) | MSE (Adaptive) | Pattern |
|------|-----------|-------|--------------|-----------------|----------------|---------|
| 200  | 0.210 | 0.201 | **0.873** | 0.564 | **0.0443** | 92% MSE reduction |
| 600  | 0.003 | 0.007 | **0.825** | 0.483 | **0.0222** | Classical near-zero |
| 1000 | 0.000 | 0.000 | **0.786** | 0.477 | **0.0168** | 97% MSE reduction |
| 1500 | 0.000 | 0.000 | **0.733** | 0.473 | **0.0140** | Strongest correction |

Weight estimation is markedly harder on ProteinGym than on ImageNet.
When the weight-learning classifier is restricted to features
independent of the annotator, coverage falls to 0.426 at n=200 and
0.008 at n=1500, and no variance-estimator variant recovers it. **On
this benchmark the supported claim is MSE reduction, not coverage.**

For detailed results, ablation studies, diagnostics and limitations,
see [`results/experiments.md`](results/experiments.md).

---

## What We Build

### Adaptive AutoEval Estimator

Adaptive AutoEval evaluates whether model performance estimates are
calibrated to the **target distribution**, not merely whether the
synthetic annotator correlates with true labels. We define the
importance-weighted PPI++ estimator:

| Variant | Description |
|---------|-------------|
| **Classical** | Labeled data only. Biased under shift. |
| **PPI++ (unweighted)** | Standard AutoEval. Biased under shift. |
| **Oracle PPI++** | PPI++ with true density ratio weights. Upper bound. |
| **Adaptive AutoEval** | PPI++ with learned importance weights. Our method. |

The weight multiplies the **full residual**:

```
mu_adapt = (λ/N) Σ_j Ê^u_j  +  (1/n) Σ_i w̃(X_i)·( φ_i − λ·Ê_i )
```

with self-normalized weights w̃, and λ* computed from **weighted**
moments Cov_w(φ, Ê) / (Var_w(Ê) + (n/N)·Var(Ê^u)). Both the point
estimate and the variance are derived from the same residual
expression. Applying the weight to φ alone is the natural-looking
alternative and is incorrect: the weight then fails to cancel against
the λÊ term, leaving a bias of λ·(E_P[Ê] − E_Q[Ê]) that does not
shrink with n, so coverage degrades as the sample grows.
`tests/test_estimator_bias.py` pins this down.

### Importance Weight Learning

We estimate the density ratio w(x) = p_target(x) / p_source(x) via
a discriminative classifier trained to distinguish labeled from
unlabeled data, with Platt scaling and probability clipping to
[ε, 1−ε]. Clipping *probabilities* to [ε, 1−ε] induces *weight*
bounds [ε/(1−ε), (1−ε)/ε] ≈ [0.0101, 99.0] at ε=0.01.

The weight-learning features must be independent of the variable that
drives the selection bias. If the discriminator is given that variable,
weight estimation becomes trivially easy and measured performance is
inflated rather than informative.

Results are insensitive to classifier choice — the spread across
logistic regression, an MLP and a random forest is under 3 points at
any n — and to ε across [0.001, 0.1]. Among alternative density-ratio
estimators, uLSIF and KLIEP underperform at every n, while KMM is
competitive on coverage but scales as O(n²). The framework accepts any
of them in place of the discriminative estimator.

### Theoretical Guarantees

We prove consistency and asymptotic normality of the weighted
estimator under mild classifier consistency conditions, and bound
the effective sample size in terms of the χ² divergence between
source and target distributions. The weight-estimation variance term
σ²_weight is O(n⁻¹) under logistic regression at the parametric rate,
hence asymptotically negligible against the two O(n⁻¹ᐟ²) terms already
included; its Fisher-information plug-in is numerically unstable and is
documented rather than shipped. See
[`adaptive_autoeval/variance.py`](adaptive_autoeval/variance.py).

### Two Domains

Adaptive AutoEval spans two evaluation domains, selected to cover
distinct mechanisms of labeling bias and real-world stakes.

| Domain | Models | Shift Mechanism | Why It Matters |
|--------|--------|-----------------|----------------|
| ImageNet | 5 ResNets | Confidence-biased sampling | Vision evaluation baseline |
| ProteinGym SPG1 | 7 foundation models | Fitness-biased annotation | Scientific ML — high stakes |

Why these two? They disentangle two drivers of misevaluation:
**selection bias** (confidence-based, ImageNet) and **experimental
bias** (fitness-based, ProteinGym). ImageNet shows that even
standard benchmark evaluation produces coverage collapse under
plausible annotator behavior. ProteinGym shows the method
generalizes to scientific domains where labeling bias is structural.

---

## Experiment Scripts

Each experiment is a self-contained Python script that loads data,
runs all estimators, scores results, and writes CSVs.

| Script | Domain | Dataset | Key Finding |
|--------|--------|---------|-------------|
| `run_extension1_imagenet.py` | ImageNet | ResNet-18/34/50/101/152 | Coverage 0.82–0.86, stable in n; MSE −48–86% |
| `run_extension1_proteingym.py` | ProteinGym | SPG1 DMS (536k variants) | 92–97% MSE reduction |
| `run_extension1_ablations.py` | ImageNet | Same as above | Robust to classifier choice and clipping |
| `run_interval_corrections.py` | Both | Same as above | Variance-estimator variants differ by 1–3 points |
| `run_baselines_and_diagnostics.py` | Both | Same as above | KMM/KLIEP/uLSIF, oracle comparison, ESS split |
| `compute_spearman.py` | ProteinGym | SPG1 DMS | Ranking analysis |
| `extract_features_imagenet.py` | ImageNet | — | ResNet-50 2048-d penultimate features |
| `extract_features_proteingym.py` | ProteinGym | — | ESM-2 640-d embeddings |
| `make_figures.py` | — | Result CSVs | Redraws every figure |

---

## Running Experiments

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

Requires Python ≥3.8; editable installs need pip ≥21.3.
All scripts use `np.random.seed(42)` for reproducibility and must be
run from the repository root, since input paths are relative to it.
No API keys required — all experiments use local preprocessed data files.

### Experiment 1: ImageNet — Synthetic Covariate Shift

```bash
python scripts/run_extension1_imagenet.py
```

Outputs to `results/imagenet/`:
- `ext1_results.csv` — coverage, MSE, ESS for all 4 estimators across n

### Experiment 2: ProteinGym — Fitness-Biased Labeling

```bash
python scripts/run_extension1_proteingym.py
```

Outputs to `results/proteingym/`:
- `ext1_pg_results.csv` — coverage, MSE, ESS across sample sizes

### Experiment 3: Ablation Studies

```bash
python scripts/run_extension1_ablations.py
```

Outputs per-setting CSVs to `results/ablations/`. The consolidated
ablation tables reported in `results/experiments.md` come from
`run_baselines_and_diagnostics.py`, which sweeps all three ablations
under a single configuration and tags each row with a config hash.

### Experiment 4: Variance-Estimator Variants

```bash
python scripts/run_interval_corrections.py
```

Outputs `interval_variants.csv` to `results/imagenet/` and
`results/proteingym/`, plus a summary report to `results/diagnostics/`.

### Experiment 5: Baselines and Diagnostics

```bash
python scripts/run_baselines_and_diagnostics.py
```

Outputs to `results/diagnostics/` (density-ratio baselines, oracle
comparison, ESS split) and corrected ablation tables to
`results/ablations/`.

### Spearman Rank Correlation Analysis

```bash
python scripts/compute_spearman.py
```

Outputs to `results/proteingym/`:
- `ext1_pg_spearman.csv` — per-trial Spearman ρ (100 trials)

### Figures

```bash
python scripts/make_figures.py
```

Reads only the CSVs under `results/`, so figures can be regenerated
without the raw data.

### Feature Extraction

```bash
python scripts/extract_features_imagenet.py
python scripts/extract_features_proteingym.py
```

The experiment scripts look for
`results/features_imagenet/resnet50_penultimate.npy` and fall back to a
documented 1-d softmax-entropy proxy when it is absent. **The results
currently in `results/` use the fallback.** Run these scripts first to
reproduce under the full feature sets. A GPU is recommended.

### Expected Runtimes

| Script | Runtime |
|--------|---------|
| `run_extension1_imagenet.py` | ~3 min |
| `run_extension1_proteingym.py` | ~10–15 min |
| `run_extension1_ablations.py` | ~10–12 min |
| `run_interval_corrections.py` | ~20 min |
| `run_baselines_and_diagnostics.py` | ~30 min |
| `compute_spearman.py` | ~5–8 min |
| `make_figures.py` | seconds |

### Tests

```bash
python -m pytest tests/ -q
```

22 tests covering the estimator form, the weighted-moment λ*, the
residual consistency of the variance, and the σ²_weight derivation.
Deselect the Monte Carlo tests with `-m "not slow"`.

---

## Repository Structure

```bash
adaptive-autoeval/
│
├── adaptive_autoeval/               # pip-installable library
│   ├── __init__.py
│   ├── estimators.py                # ppi_unweighted, ppi_weighted
│   ├── weights.py                   # learn_importance_weights
│   └── variance.py                  # σ²_weight derivation, bootstrap
│
├── scripts/                         # Experiment scripts
│   ├── run_extension1_imagenet.py
│   ├── run_extension1_proteingym.py
│   ├── run_extension1_ablations.py
│   ├── run_interval_corrections.py
│   ├── run_baselines_and_diagnostics.py
│   ├── compute_spearman.py
│   ├── extract_features_imagenet.py
│   ├── extract_features_proteingym.py
│   └── make_figures.py
│
├── tests/
│   ├── test_estimator_bias.py       # estimator form, λ*, residual consistency
│   └── test_variance.py             # σ²_weight derivation
│
├── results/
│   ├── experiments.md               # Cross-domain results summary
│   ├── imagenet/                    # main table, interval variants, figures
│   ├── proteingym/                  # main table, variants, Spearman, figures
│   ├── ablations/                   # severity, classifier, clipping
│   └── diagnostics/                 # DRE baselines, oracle, ESS split
│
├── data/
│   ├── imagenet/README.md           # Download instructions
│   └── proteingym/README.md         # Download instructions
│
├── pyproject.toml
├── requirements.txt
├── LICENSE                          # MIT (code) + CC-BY-4.0 (data)
└── README.md

```

---

## Models Tested

| Model Family | Models | Domain | Performance |
|-------------|--------|--------|-------------|
| ResNet | ResNet-18, 34, 50, 101, 152 | ImageNet | Adaptive coverage 0.82–0.86 across n |
| Protein LMs | CARP, ESM-1b, ESM-1v, ESM-2 | ProteinGym | 92–97% MSE reduction |
| Protein LMs | ProGen2, RITA, UniRep | ProteinGym | Adaptive coverage 0.73–0.87 across n |

Annotator: ResNet-101 confidence scores (ImageNet),
VESPA conservation scores (ProteinGym).

---

## Data Setup

Raw input arrays are not tracked in this repository.

**ImageNet**: See [`data/imagenet/README.md`](data/imagenet/README.md).
After setup, place preprocessed numpy files at:

results/phi_imagenet/resnet{18,34,50,101,152}.npy

results/synthetic_imagenet/resnet{18,34,50,101,152}.npy

**ProteinGym**: See [`data/proteingym/README.md`](data/proteingym/README.md).
After downloading from https://proteingym.org, place files at:

data/proteingym/SPG1_STRSG_Olson_2014.csv

data/proteingym/SPG1_STRSG_Olson_2014_zero_shot.csv

### Reproducibility note

Experiments were run on a consumer laptop (AMD Ryzen 7 5800H, 8 GB RAM,
CPU only) under Windows 11 with a Conda Python 3.10 environment. The
ablation scripts draw the unlabeled subsample from the global NumPy
random stream rather than a per-trial seeded generator; each script as a
whole is deterministic at `seed=42`, but individual trials are not
reproducible in isolation and fresh runs may drift by up to 0.001.

---

## Citation

If you use Adaptive AutoEval in your research, please cite:

```bibtex
@misc{adaptive_autoeval_2026,
  title  = {AutoEval under Unknown Covariate Shift: Learning
            Importance Weights from Unlabeled Data},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

---

## License

Code: MIT License.
Data and results: CC-BY-4.0.
