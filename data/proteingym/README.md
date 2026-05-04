# ProteinGym Data Setup

## Overview

This project uses the ProteinGym DMS benchmark (SPG1) to evaluate Adaptive AutoEval under fitness-biased labeling.

The dataset contains:

- Experimental fitness measurements (ground truth)
- Zero-shot model predictions (synthetic annotator)

Used in:

```bash
python scripts/run_extension1_proteingym.py
```

Download Instructions

Download from the official ProteinGym website:

https://www.proteingym.org/

Required files:

SPG1_STRSG_Olson_2014.csv

SPG1_STRSG_Olson_2014_zero_shot.csv


Required Directory Structure:

Place files exactly here:

```bash
data/proteingym/
├── SPG1_STRSG_Olson_2014.csv
├── SPG1_STRSG_Olson_2014_zero_shot.csv
```

This path is expected by the experiment scripts.
