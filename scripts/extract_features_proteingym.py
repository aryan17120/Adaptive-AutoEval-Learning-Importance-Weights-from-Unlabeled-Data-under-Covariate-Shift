"""
scripts/extract_features_proteingym.py
─────────────────────────────────────────────────────────────────────────────
Extract ESM-2 mean-pooled sequence embeddings for every variant in the SPG1
ProteinGym assay:

  results/features_proteingym/esm2_embeddings.npy   shape (N_variants, 640)

These features are what the weight-learning classifier consumes. They must
encode sequence identity without encoding the fitness label: in this assay the
normalised fitness drives both the selection bias and the VESPA annotator, so
a feature set built from calibrated model scores lets the discriminator
separate labeled from unlabeled trivially, inflating measured performance.

WHY THE VARIABLE WINDOW, NOT THE FULL CONSTRUCT
───────────────────────────────────────────────
Every `mutated_sequence` in this assay is 448 residues, and every variant
carries one or two substitutions confined to a narrow window (positions
228-282 for SPG1). The remaining ~390 residues are byte-identical across all
536,962 variants.

Mean-pooling a 448-residue embedding would therefore average two changed
residues against ~446 constant ones, leaving variant-to-variant differences
near the numerical noise floor. Restricting the encoder to the variable window
(plus a flank) keeps all of the variant-discriminating signal, removes a
constant offset that carries none, and is ~6x cheaper. The window is derived
from the data, not hard-coded -- see `variable_window()`.

MODEL
─────
Default `esm2_t30_150M_UR50D`, whose representation dimension is 640 and which
therefore matches the documented output shape. `--model` accepts any fair-esm
checkpoint; note that `esm2_t33_650M_UR50D` emits 1280-d, not 640-d.

USAGE
─────
  python scripts/extract_features_proteingym.py \\
      --csv_path data/proteingym/SPG1_STRSG_Olson_2014_zero_shot.csv \\
      [--out_dir results/features_proteingym] \\
      [--model esm2_t30_150M_UR50D] \\
      [--batch_size 128] [--device cuda] [--fp16] [--limit N]

REQUIREMENTS
────────────
  pip install fair-esm torch

OUTPUT
──────
  results/features_proteingym/esm2_embeddings.npy
    float32 array, shape (N_variants, embed_dim)
    Row i corresponds to row i of the CSV, read in file order, matching the
    indexing used by scripts/run_extension1_proteingym.py.
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd


def variable_window(mutants, flank=8):
    """Infer the mutated-position window from the mutant strings."""
    lo, hi = None, None
    for m in mutants:
        for tok in str(m).split(":"):
            digits = re.findall(r"\d+", tok)
            if not digits:
                continue
            pos = int(digits[0])
            lo = pos if lo is None else min(lo, pos)
            hi = pos if hi is None else max(hi, pos)
    if lo is None:
        raise SystemExit("Could not parse any mutated positions.")
    return lo, hi, flank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--out_dir", default="results/features_proteingym")
    ap.add_argument("--model", default="esm2_t30_150M_UR50D")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true",
                    help="Half precision; roughly 2x faster and halves VRAM")
    ap.add_argument("--flank", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    try:
        import torch
        import esm
    except ImportError as e:
        raise ImportError("pip install fair-esm torch") from e

    df = pd.read_csv(args.csv_path, usecols=["mutant", "mutated_sequence"])
    if args.limit:
        df = df.iloc[:args.limit]
    seqs = df["mutated_sequence"].tolist()
    print(f"Variants: {len(seqs)}   full length: {len(seqs[0])}")

    lo, hi, flank = variable_window(df["mutant"], args.flank)
    # Mutant positions are 1-indexed into the construct.
    start = max(0, lo - 1 - flank)
    stop = min(len(seqs[0]), hi + flank)
    print(f"Mutated positions {lo}-{hi}; embedding window [{start}:{stop}] "
          f"= {stop - start} residues")
    windows = [s[start:stop] for s in seqs]

    n_const = len(set(windows))
    print(f"Distinct windows: {n_const} / {len(windows)}")

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    model, alphabet = getattr(esm.pretrained, args.model)()
    n_layers = model.num_layers
    model = model.eval().to(device)
    if args.fp16 and device.type == "cuda":
        model = model.half()
    torch.set_grad_enabled(False)

    bc = alphabet.get_batch_converter()
    dim = model.embed_dim
    print(f"Model {args.model}: {n_layers} layers, {dim}-d, device {device}, "
          f"fp16={args.fp16 and device.type == 'cuda'}")

    out = np.empty((len(windows), dim), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(windows), args.batch_size):
        chunk = windows[i:i + args.batch_size]
        _, _, toks = bc([(str(j), s) for j, s in enumerate(chunk)])
        toks = toks.to(device)
        rep = model(toks, repr_layers=[n_layers])["representations"][n_layers]
        # Mean-pool over real residues only (drop BOS/EOS and padding).
        mask = (toks != alphabet.padding_idx)
        mask[:, 0] = False
        for r in range(len(chunk)):
            last = int(mask[r].nonzero()[-1])
            mask[r, last] = False
        m = mask.unsqueeze(-1).to(rep.dtype)
        pooled = (rep * m).sum(1) / m.sum(1).clamp(min=1)
        out[i:i + len(chunk)] = pooled.float().cpu().numpy()

        done = i + len(chunk)
        if (i // args.batch_size) % 20 == 0 or done == len(windows):
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f"  {done:7d}/{len(windows)}  {rate:7.1f} seq/s  "
                  f"elapsed {el/60:5.1f}m  eta "
                  f"{(len(windows)-done)/max(rate,1e-9)/60:5.1f}m", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, "esm2_embeddings.npy")
    np.save(dest, out)
    print(f"Saved {out.shape} -> {dest}")


if __name__ == "__main__":
    main()
