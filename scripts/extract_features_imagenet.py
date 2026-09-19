"""
scripts/extract_features_imagenet.py
─────────────────────────────────────────────────────────────────────────────
Extract ResNet-50 2048-d penultimate-layer (avgpool) features for the 50k
ImageNet validation images:

  results/features_imagenet/resnet50_penultimate.npy   shape (50000, 2048)

These features are what the weight-learning classifier consumes. They must be
informative about image content without being the annotator score itself: if
the discriminator is handed the scalar that generated the selection bias
(here the ResNet-101 confidence), weight estimation becomes trivially easy and
measured performance is inflated rather than informative.

ROW ORDER -- IMPORTANT
──────────────────────
Row i of the output must correspond to row i of
``results/phi_imagenet/*.npy`` and ``results/synthetic_imagenet/*.npy``.
Those arrays were produced by iterating the validation set in **filename
order** (ILSVRC2012_val_00000001.JPEG ... ILSVRC2012_val_00050000.JPEG) with
``shuffle=False``.

This script reproduces that order exactly. It does NOT use
``torchvision.datasets.ImageFolder``, which sorts by class directory and would
silently misalign every row.

To confirm alignment after extraction, reapply ResNet-50's own ``fc`` head to
the saved features and check that the resulting per-image correctness matches
``results/phi_imagenet/resnet50.npy``.

PREPROCESSING
─────────────
Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet mean/std),
matching the pipeline that produced the phi/synthetic arrays.

USAGE
─────
  python scripts/extract_features_imagenet.py \\
      --val_dir /path/to/ILSVRC2012/val \\
      [--map_file /path/to/val_map.txt] \\
      [--out_dir results/features_imagenet] \\
      [--batch_size 64] [--device cpu] [--limit N]

  --map_file  optional; a text file whose first whitespace-separated token per
              line is the image filename, in the intended order. When omitted,
              the script globs val_dir for *.JPEG and sorts by filename, which
              gives the same order for a standard flat validation directory.

REQUIREMENTS
────────────
  pip install torch torchvision

OUTPUT
──────
  results/features_imagenet/resnet50_penultimate.npy
    float32 array, shape (n_images, 2048)
"""

import argparse
import glob
import os
import time

import numpy as np


_TF = None


def _transform():
    """Preprocessing matching the pipeline that produced the phi arrays."""
    global _TF
    if _TF is None:
        from torchvision import transforms
        _TF = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return _TF


class FlatValDataset:
    """Module-level so Windows `spawn` workers can pickle it."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return _transform()(Image.open(self.paths[i]).convert("RGB"))


def build_file_list(val_dir, map_file):
    """Return image paths in the order the phi/synthetic arrays used."""
    if map_file:
        with open(map_file) as f:
            names = [ln.split()[0] for ln in f if ln.strip()]
    else:
        names = sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(val_dir, "*.JPEG"))
        )
    if not names:
        raise SystemExit(f"No images found in {val_dir}")
    return [os.path.join(val_dir, n) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True,
                    help="Directory holding the 50k validation JPEGs (flat)")
    ap.add_argument("--map_file", default=None,
                    help="Optional ordering file; first token per line is the filename")
    ap.add_argument("--out_dir", default="results/features_imagenet")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N images (for benchmarking)")
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import models
    except ImportError as e:
        raise ImportError(
            "torch and torchvision are required: pip install torch torchvision"
        ) from e

    paths = build_file_list(args.val_dir, args.map_file)
    if args.limit:
        paths = paths[:args.limit]
    print(f"Images to process: {len(paths)}")
    print(f"  first: {os.path.basename(paths[0])}")
    print(f"  last:  {os.path.basename(paths[-1])}")

    loader = DataLoader(FlatValDataset(paths), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()          # avgpool output -> 2048-d
    net.eval().to(device)
    torch.set_grad_enabled(False)

    out = np.empty((len(paths), 2048), dtype=np.float32)
    pos, t0 = 0, time.time()
    for bi, batch in enumerate(loader):
        feats = net(batch.to(device)).cpu().numpy()
        out[pos:pos + len(feats)] = feats
        pos += len(feats)
        if bi % 20 == 0 or pos == len(paths):
            el = time.time() - t0
            rate = pos / max(el, 1e-9)
            eta = (len(paths) - pos) / max(rate, 1e-9)
            print(f"  {pos:6d}/{len(paths)}  {rate:6.1f} img/s  "
                  f"elapsed {el/60:5.1f}m  eta {eta/60:5.1f}m", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, "resnet50_penultimate.npy")
    np.save(dest, out)
    print(f"Saved {out.shape} -> {dest}")


if __name__ == "__main__":
    main()
