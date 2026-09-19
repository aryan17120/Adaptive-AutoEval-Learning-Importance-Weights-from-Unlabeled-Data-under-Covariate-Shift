"""
scripts/extract_features_proteingym.py
─────────────────────────────────────────────────────────────────────────────
Extract ESM-2 (650M) 640-d mean-pooled embeddings for all protein
variants in the SPG1 ProteinGym assay and save to
  results/features_proteingym/esm2_embeddings.npy   shape (N_variants, 640)

These replace any setup where the feature matrix includes
calibrated model scores and the VESPA annotator score. Under that setup the
discriminator was trained on features that correlate directly with the shift
variable (fitness Y_norm drives both the selection bias and the annotator
predictions), making weight estimation circular.

The ESM-2 640-d sequence embedding provides a representation that encodes
protein sequence information independently of the fitness label, breaking
the circular coupling.

USAGE
─────
  # From SRC/ directory:
  python scripts/extract_features_proteingym.py \\
      --csv_path data/proteingym/SPG1_STRSG_Olson_2014_zero_shot.csv \\
      --out_dir  results/features_proteingym \\
      [--batch_size 64] [--device cuda]

REQUIREMENTS
────────────
  pip install fair-esm torch
  Or: pip install esm  (newer API)

OUTPUT
──────
  results/features_proteingym/esm2_embeddings.npy
    float32 array, shape (N_variants, 640)
    Row i corresponds to row i of the filtered (non-NaN) SPG1 CSV,
    matching the pool_idx indexing used in run_extension1_proteingym.py.

NOTE
────
For SPG1 with 536,962 variants, embedding extraction is compute-intensive.
A single A100 GPU requires approximately 2-4 hours.
Alternatively, pre-computed ESM-2 embeddings for ProteinGym assays are
available from the ProteinGym repository:
  https://github.com/OATML-Markslab/ProteinGym
"""

import argparse, os
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",   required=True,
                        help="Path to SPG1_STRSG_Olson_2014_zero_shot.csv")
    parser.add_argument("--out_dir",    default="results/features_proteingym")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    try:
        import torch
        import esm
    except ImportError:
        raise ImportError(
            "fair-esm and torch are required. "
            "Install with: pip install fair-esm torch\n"
            "Or see: https://github.com/facebookresearch/esm"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    print(f"Loading CSV: {args.csv_path} ...")
    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["DMS_score"]).reset_index(drop=True)
    print(f"  {len(df)} non-NaN variants")

    if "mutant_sequence" not in df.columns and "sequence" not in df.columns:
        raise ValueError(
            "CSV must have a 'mutant_sequence' or 'sequence' column. "
            "Check the ProteinGym file format."
        )
    seq_col = "mutant_sequence" if "mutant_sequence" in df.columns else "sequence"
    sequences = df[seq_col].tolist()

    print("Loading ESM-2 (650M) model ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    all_embeddings = []
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for i in range(n_batches):
            batch_seqs = sequences[i * args.batch_size:(i + 1) * args.batch_size]
            data = [(f"seq_{j}", s) for j, s in
                    enumerate(batch_seqs, start=i * args.batch_size)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            results = model(tokens, repr_layers=[33], return_contacts=False)
            # Mean-pool over sequence positions (exclude BOS/EOS)
            emb = results["representations"][33]   # (batch, L+2, 640)
            emb = emb[:, 1:-1, :].mean(dim=1)      # (batch, 640)
            all_embeddings.append(emb.cpu().numpy().astype(np.float32))
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{n_batches}  "
                      f"({min((i+1)*args.batch_size, len(sequences))}/{len(sequences)} variants)")

    embeddings = np.concatenate(all_embeddings, axis=0)   # (N, 640)
    out_path = os.path.join(args.out_dir, "esm2_embeddings.npy")
    np.save(out_path, embeddings)
    print(f"\nSaved: {out_path}  shape={embeddings.shape}  dtype={embeddings.dtype}")
    print("Run run_extension1_proteingym.py — it will auto-detect and use these features.")


if __name__ == "__main__":
    main()
