"""Run MiDaS monocular depth estimation on ECUSTFD test images.

Saves depth maps (HWC uint8 PNGs) to runs/midas_depth_maps/ so they can
be reused without re-running the model.

Usage
-----
    python scripts/midas_depth_inference.py \
        --source datasets/ECUSTFD/images/test \
        --model MiDaS_small \
        --output runs/midas_depth_maps \
        --max-images 0

Output layout
------------
runs/midas_depth_maps/
  {image_stem}.png   # single-channel depth map (higher = farther)
  summary.json       # image count, time, model info
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/ECUSTFD/images/test"),
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MiDaS_small",
        choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"],
        help="Which depth model to load from torch.hub.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/midas_depth_maps"),
        help="Directory to save depth PNG maps.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images to process (0 = all).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=384,
        help="Inference image size for MiDaS.",
    )
    return parser.parse_args()


def collect_images(source: Path, max_images: int) -> list[Path]:
    images = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix in IMAGE_EXTS
    )
    if max_images > 0 and len(images) > max_images:
        return images[:max_images]
    return images


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load MiDaS
    print(f"[midas] Loading {args.model} ...")
    t0 = time.time()
    import torch
    torch.hub.set_dir(".cache")
    model = torch.hub.load("intel-isl/Midas", args.model, trust_repo=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    load_time = time.time() - t0
    print(f"[midas] Loaded in {load_time:.1f}s on {device}")

    # Collect images
    images = collect_images(args.source, args.max_images)
    print(f"[midas] Processing {len(images)} images ...")

    import numpy as np
    from PIL import Image
    import cv2

    t1 = time.time()
    n_saved = 0
    n_failed = 0

    for img_path in images:
        try:
            # Load and resize image
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)

            # MiDaS expects 384x384 input
            img_resized = cv2.resize(img_np, (args.imgsz, args.imgsz),
                                    interpolation=cv2.INTER_LINEAR)

            # Normalise to [0,1] as MiDaS expects
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std

            with torch.no_grad():
                depth = model(img_tensor.unsqueeze(0).to(device))
                depth = depth.squeeze().cpu().numpy()

            # Normalise depth to 0-255 uint8 for storage
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
            # Also save raw float32 as .npy for calorie computation
            out_stem = img_path.stem
            out_png = args.output / f"{out_stem}_depth.png"
            out_npy = args.output / f"{out_stem}_depth.npy"

            cv2.imwrite(str(out_png), depth_norm)
            np.save(str(out_npy), depth.astype(np.float32))

            n_saved += 1
            if n_saved % 100 == 0:
                elapsed = time.time() - t1
                print(f"[midas] {n_saved}/{len(images)} ({n_saved/max(elapsed,1e-6):.1f} img/s)")

        except Exception as exc:
            print(f"[midas] FAIL {img_path.name}: {exc}")
            n_failed += 1

    elapsed = time.time() - t1
    print(f"[midas] Done: {n_saved} saved, {n_failed} failed in {elapsed:.1f}s ({n_saved/max(elapsed,1e-6):.1f} img/s)")

    # Save summary
    summary = {
        "model": args.model,
        "device": device,
        "load_time_s": round(load_time, 1),
        "n_saved": n_saved,
        "n_failed": n_failed,
        "total_time_s": round(elapsed, 1),
        "fps": round(n_saved / max(elapsed, 1e-6), 2),
        "imgsz": args.imgsz,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[midas] Wrote summary to {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
