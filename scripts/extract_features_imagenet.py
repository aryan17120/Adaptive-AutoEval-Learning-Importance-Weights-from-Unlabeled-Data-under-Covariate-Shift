"""
scripts/extract_features_imagenet.py
─────────────────────────────────────────────────────────────────────────────
Extract ResNet-50 2048-d penultimate layer features for all 50k
ImageNet validation images and save to
  results/features_imagenet/resnet50_penultimate.npy   shape (50000, 2048)

These replace any setup where `features = syn_all.copy()`
fed the annotator score matrix directly to the weight-learning classifier.
Under that setup the discriminator received the exact scalar that generated
the selection bias (ResNet-101 confidence = syn_all[:,3] = shift variable),
making the learned weights trivially good — not because the method works,
but because the classifier is cheating.

The ResNet-50 penultimate layer (avgpool output) provides a 2048-d
representation that is informative about image content without being the
annotator score itself.

USAGE
─────
  # From SRC/ directory:
  python scripts/extract_features_imagenet.py \\
      --val_dir /path/to/ILSVRC2012/val \\
      --out_dir results/features_imagenet \\
      [--batch_size 256] [--device cuda]

REQUIREMENTS
────────────
  pip install torch torchvision
  ImageNet validation set at val_dir, with one subfolder per class (1000 classes).
  Images must be in the same order as the phi_imagenet/ .npy files, i.e. sorted
  by (class_idx, filename) — the standard torchvision ImageFolder order.

OUTPUT
──────
  results/features_imagenet/resnet50_penultimate.npy
    float32 array, shape (50000, 2048)
    Row i corresponds to image i in phi_imagenet/resnet{18,34,...}.npy
"""

import argparse, os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir",    required=True,
                        help="Path to ImageNet val/ directory (1000 subfolders)")
    parser.add_argument("--out_dir",    default="results/features_imagenet")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    try:
        import torch
        import torchvision
        from torchvision import transforms, datasets, models
    except ImportError:
        raise ImportError(
            "torch and torchvision are required. "
            "Install with: pip install torch torchvision"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    # Preprocessing: standard ImageNet normalisation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading ImageNet val from {args.val_dir} ...")
    dataset = datasets.ImageFolder(args.val_dir, transform=transform)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda")
    )
    print(f"  {len(dataset)} images, {len(loader)} batches")

    # ResNet-50 with pretrained weights; remove final FC layer
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Hook the avgpool output (penultimate, 2048-d)
    model.fc = torch.nn.Identity()        # replace final FC with identity
    model = model.to(device)
    model.eval()

    all_feats = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            feats = model(imgs)            # (batch, 2048)
            all_feats.append(feats.cpu().numpy().astype(np.float32))
            if (i + 1) % 50 == 0:
                n_done = min((i + 1) * args.batch_size, len(dataset))
                print(f"  {n_done}/{len(dataset)} images processed")

    features = np.concatenate(all_feats, axis=0)    # (50000, 2048)
    out_path = os.path.join(args.out_dir, "resnet50_penultimate.npy")
    np.save(out_path, features)
    print(f"\nSaved: {out_path}  shape={features.shape}  dtype={features.dtype}")
    print("Run run_extension1_imagenet.py — it will auto-detect and use these features.")


if __name__ == "__main__":
    main()
