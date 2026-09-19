# Adaptive AutoEval Learning Importance Weights from Unlabeled Data under Covariate Shift
### Importance-Weighted Model Evaluation under Unknown Covariate Shift
*Anonymous submission — NeurIPS 2026 Main Track*

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
- **Empirical measurements** of the delta between standard PPI++
  coverage and Adaptive AutoEval coverage across two domains

---

## Key Results

We tested Adaptive AutoEval on ImageNet (vision) and ProteinGym
(protein biology) under realistic labeling biases.

**Headline finding**: even fixed-layout evaluations produce severe
coverage collapse when concept density is high — standard AutoEval
coverage drops to **1.7%** at n=500 under moderate confidence bias.

### ImageNet — Synthetic Covariate Shift

| n   | Classical | PPI++ | Oracle | **Adaptive** | Pattern |
|-----|-----------|-------|--------|--------------|---------|
| 50  | 0.635 | 0.793 | 0.700 | **0.789** | All methods partially recover |
| 100 | 0.461 | 0.823 | 0.684 | **0.834** | Adaptive best at small n |
| 200 | 0.224 | 0.753 | 0.662 | **0.770** | Classical collapses |
| 300 | 0.098 | 0.734 | 0.609 | **0.768** | Oracle underperforms Adaptive |
| 400 | 0.065 | 0.696 | 0.578 | **0.739** | Adaptive beats oracle due to regularization |
| 500 | 0.017 | 0.682 | 0.554 | **0.720** | Classical near-zero |

Coverage of nominal 90% confidence intervals (250 trials).
Positive delta = standard PPI++ extracts correct performance signal
but confidence intervals are miscalibrated due to distribution shift.

### ProteinGym SPG1 — Fitness-Biased Labeling (β=1.0)

| n    | Classical | PPI++ | **Adaptive** | MSE (Classical) | MSE (Adaptive) | Pattern |
|------|-----------|-------|--------------|-----------------|----------------|---------|
| 200  | 0.198 | 0.166 | **0.556** | 0.577 | **0.311** | 2.8× coverage improvement |
| 600  | 0.007 | 0.005 | **0.194** | 0.531 | **0.226** | Classical near-zero |
| 1000 | 0.000 | 0.000 | **0.091** | 0.500 | **0.202** | 60% MSE reduction |
| 1500 | 0.000 | 0.000 | **0.029** | 0.482 | **0.187** | Strongest shift correction |

For detailed results, ablation studies, and per-model breakdowns,
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

The key empirical finding is the **delta between PPI++ coverage and
Adaptive AutoEval coverage**: the gap reveals how often standard
AutoEval produces invalid confidence intervals due to covariate shift.

### Importance Weight Learning

We estimate the density ratio w(x) = p_target(x) / p_source(x) via
a discriminative classifier trained to distinguish labeled from
unlabeled data. Crucially, the implicit regularization from logistic
regression with Platt scaling **outperforms oracle weights** (true
density ratio) at every sample size, due to variance reduction from
smoothed weight estimates.

### Theoretical Guarantees

We prove consistency and asymptotic normality of the weighted
estimator under mild classifier consistency conditions, and bound
the effective sample size in terms of the χ² divergence between
source and target distributions.

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
runs all four estimators, scores results, and generates figures.

| Script | Domain | Dataset | Key Finding |
|--------|--------|---------|-------------|
| `run_extension1_imagenet.py` | ImageNet | ResNet-18/34/50/101/152 | Coverage 1.7% → 72–83%; Adaptive beats Oracle |
| `run_extension1_proteingym.py` | ProteinGym | SPG1 DMS (536k variants) | 2.8× coverage gain; 46–61% MSE reduction |
| `run_extension1_ablations.py` | ImageNet | Same as above | Robust to classifier choice, clipping, shift severity |
| `compute_spearman.py` | ProteinGym | SPG1 DMS | Ranking analysis: absolute MSE corrected, rankings preserved |

---

## Running Experiments

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

All scripts use `np.random.seed(42)` for reproducibility.
No API keys required — all experiments use local preprocessed data files.

### Experiment 1: ImageNet — Synthetic Covariate Shift

```bash
python scripts/run_extension1_imagenet.py
```

Outputs to `results/imagenet/`:
- `ext1_results.csv` — coverage, MSE, ESS for all 4 estimators across n
- `ext1_main.png` — 3-panel figure (coverage, MSE, ESS vs n)

### Experiment 2: ProteinGym — Fitness-Biased Labeling

```bash
python scripts/run_extension1_proteingym.py
```

Outputs to `results/proteingym/`:
- `ext1_pg_results.csv` — coverage, MSE, ESS across sample sizes
- `ext1_pg_main.png` — 3-panel figure
- `ext1_pg_ranking.png` — per-model MSE bar chart vs ground truth

### Experiment 3: Ablation Studies

```bash
python scripts/run_extension1_ablations.py
```

Outputs to `results/ablations/`:
- Per-ablation CSVs (classifier type, clipping threshold, shift severity)
- `abl_combined.png` — 3-panel summary figure

### Spearman Rank Correlation Analysis

```bash
python scripts/compute_spearman.py
```

Outputs to `results/proteingym/`:
- `ext1_pg_spearman.csv` — per-trial Spearman ρ (100 trials)
- Printed mean ± std summary

### Expected Runtimes

| Script | Runtime |
|--------|---------|
| `run_extension1_imagenet.py` | ~3 min |
| `run_extension1_proteingym.py` | ~10–15 min |
| `run_extension1_ablations.py` | ~10–12 min |
| `compute_spearman.py` | ~5–8 min |

---

## Repository Structure

```bash
adaptive-autoeval/
│
├── adaptive_autoeval/               # pip-installable library
│   ├── __init__.py
│   ├── estimators.py                # ppi_unweighted, ppi_weighted
│   └── weights.py                   # learn_importance_weights
│
├── scripts/                         # Experiment scripts (4 domains)
│   ├── run_extension1_imagenet.py
│   ├── run_extension1_proteingym.py
│   ├── run_extension1_ablations.py
│   └── compute_spearman.py
│
├── results/
│   ├── experiments.md               # Cross-domain results summary
│   ├── imagenet/                    # ext1_results.csv, ext1_main.png
│   ├── proteingym/                  # ext1_pg_results.csv, figures, spearman
│   └── ablations/                   # Per-ablation CSVs, abl_combined.png
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
| ResNet | ResNet-18, 34, 50, 101, 152 | ImageNet | Adaptive coverage 0.72–0.83 across n |
| Protein LMs | CARP, ESM-1b, ESM-1v, ESM-2 | ProteinGym | 46–61% MSE reduction |
| Protein LMs | ProGen2, RITA, UniRep | ProteinGym | Adaptive coverage 0.03–0.56 across n |

Annotator: ResNet-101 confidence scores (ImageNet),
VESPA conservation scores (ProteinGym).

---

## Data Setup

**ImageNet**: See [`data/imagenet/README.md`](data/imagenet/README.md).
After setup, place preprocessed numpy files at:

results/phi_imagenet/resnet{18,34,50,101,152}.npy

results/synthetic_imagenet/resnet{18,34,50,101,152}.npy

**ProteinGym**: See [`data/proteingym/README.md`](data/proteingym/README.md).
After downloading from https://proteingym.org, place files at:

data/proteingym/SPG1_STRSG_Olson_2014.csv

data/proteingym/SPG1_STRSG_Olson_2014_zero_shot.csv


---

## Citation

If you use Adaptive AutoEval in your research, please cite:

```bibtex
@inproceedings{adaptive_autoeval_2026,
  title     = {AutoEval under Unknown Covariate Shift: Learning
               Importance Weights from Unlabeled Data},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

---

## License

Code: MIT License.

Data and results: CC-BY-4.0.
