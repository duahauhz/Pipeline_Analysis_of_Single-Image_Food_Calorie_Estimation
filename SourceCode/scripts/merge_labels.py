"""
Merge pseudo-labels với ground truth labels.
Strategy: confidence-weighted mix
- Nếu pseudo-label confidence > threshold (0.85): giữ nguyên bbox của pseudo (tin tưởng teacher)
- Nếu confidence thấp: dùng GT bbox
- Nếu pseudo phát hiện object mà GT không có: thêm vào (teacher thấy gì mới thêm)
- Nếu GT có mà pseudo không: giữ GT
"""
import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent
GT_LABELS_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "labels" / "train"
PSEUDO_LABELS_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "labels" / "train_pseudo_v13"
MERGED_LABELS_DIR = PROJECT_ROOT / "datasets" / "ECUSTFD" / "labels" / "train_distilled"

# Strategy:
# 0.7*conf_pseudo + 0.3*gt (for cases where both exist) = soft label
# If only GT: keep GT
# If only pseudo (high conf > 0.85): add pseudo
# If pseudo conf low: use GT only

CONF_THRESHOLD = 0.5  # Minimum confidence to trust teacher


def merge_one_label(gt_path, pseudo_path, merged_path):
    """Merge labels for a single image.

    YOLO format: cls cx cy w h [conf]
    - GT format: cls cx cy w h
    - Pseudo format: cls cx cy w h conf
    """
    # Read GT
    gt_boxes = []
    if gt_path.exists():
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    gt_boxes.append({
                        "cls": int(parts[0]),
                        "cx": float(parts[1]),
                        "cy": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4]),
                        "conf": 1.0,  # GT is ground truth = full confidence
                        "source": "gt"
                    })

    # Read pseudo
    pseudo_boxes = []
    if pseudo_path.exists():
        with open(pseudo_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    pseudo_boxes.append({
                        "cls": int(parts[0]),
                        "cx": float(parts[1]),
                        "cy": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4]),
                        "conf": float(parts[5]),
                        "source": "pseudo"
                    })

    # Match GT and pseudo by IoU > 0.5
    def iou(b1, b2):
        # b1, b2 are dicts with cx, cy, w, h
        x1_min = b1["cx"] - b1["w"]/2
        y1_min = b1["cy"] - b1["h"]/2
        x1_max = b1["cx"] + b1["w"]/2
        y1_max = b1["cy"] + b1["h"]/2
        x2_min = b2["cx"] - b2["w"]/2
        y2_min = b2["cy"] - b2["h"]/2
        x2_max = b2["cx"] + b2["w"]/2
        y2_max = b2["cy"] + b2["h"]/2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        inter = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = b1["w"] * b1["h"]
        area2 = b2["w"] * b2["h"]
        union = area1 + area2 - inter
        return inter / (union + 1e-9)

    matched_gt = set()
    matched_pseudo = set()
    pairs = []  # (gt_idx, pseudo_idx)

    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pseudo_boxes):
            if pi in matched_pseudo:
                continue
            if gb["cls"] == pb["cls"] and iou(gb, pb) > 0.5:
                pairs.append((gi, pi))
                matched_gt.add(gi)
                matched_pseudo.add(pi)
                break

    merged_boxes = []

    # Process matched pairs: weighted average of GT + pseudo bbox
    for gi, pi in pairs:
        gb = gt_boxes[gi]
        pb = pseudo_boxes[pi]
        # Weight by confidence
        w_pseudo = pb["conf"]
        w_gt = 1.0
        total = w_pseudo + w_gt
        merged_boxes.append({
            "cls": gb["cls"],
            "cx": (gb["cx"] * w_gt + pb["cx"] * w_pseudo) / total,
            "cy": (gb["cy"] * w_gt + pb["cy"] * w_pseudo) / total,
            "w": (gb["w"] * w_gt + pb["w"] * w_pseudo) / total,
            "h": (gb["h"] * w_gt + pb["h"] * w_pseudo) / total,
            "conf": max(gb["conf"], pb["conf"]),
            "source": "merged"
        })

    # Add unmatched GT (pseudo missed it) — but only if pseudo's confidence on same class area is low
    for gi, gb in enumerate(gt_boxes):
        if gi not in matched_gt:
            merged_boxes.append(gb)

    # Add high-confidence unmatched pseudo (new detection from teacher)
    for pi, pb in enumerate(pseudo_boxes):
        if pi not in matched_pseudo and pb["conf"] > CONF_THRESHOLD:
            # Only add if not duplicate of any existing merged box
            is_dup = False
            for mb in merged_boxes:
                if mb["cls"] == pb["cls"] and iou(mb, pb) > 0.3:
                    is_dup = True
                    break
            if not is_dup:
                merged_boxes.append(pb)

    # Write merged
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w") as f:
        for b in merged_boxes:
            f.write(f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

    return len(merged_boxes), len(gt_boxes), len(pseudo_boxes)


def main():
    print("=" * 70)
    print("Merge pseudo-labels (YOLOv13n) with ground truth")
    print("=" * 70)
    print(f"GT dir: {GT_LABELS_DIR}")
    print(f"Pseudo dir: {PSEUDO_LABELS_DIR}")
    print(f"Merged dir: {MERGED_LABELS_DIR}")
    print(f"Conf threshold for unmatched pseudo: {CONF_THRESHOLD}")
    print()

    # Find all GT labels
    gt_files = sorted(GT_LABELS_DIR.glob("*.txt"))
    print(f"Found {len(gt_files)} GT label files")

    # Stats
    n_total = 0
    n_with_pseudo = 0
    total_gt_boxes = 0
    total_pseudo_boxes = 0
    total_merged_boxes = 0
    n_merged_added = 0
    n_merged_pseudo_only = 0

    for gt_path in gt_files:
        pseudo_path = PSEUDO_LABELS_DIR / gt_path.name
        merged_path = MERGED_LABELS_DIR / gt_path.name

        n_merged, n_gt, n_pseudo = merge_one_label(gt_path, pseudo_path, merged_path)

        n_total += 1
        total_gt_boxes += n_gt
        total_pseudo_boxes += n_pseudo
        total_merged_boxes += n_merged

        if pseudo_path.exists():
            n_with_pseudo += 1

        if n_merged > n_gt:
            n_merged_added += 1
        if n_pseudo > 0 and n_gt == 0:
            n_merged_pseudo_only += 1

    print(f"\n[Stats] Processed: {n_total} images")
    print(f"[Stats] With pseudo-labels: {n_with_pseudo}")
    print(f"[Stats] Total GT boxes: {total_gt_boxes}")
    print(f"[Stats] Total pseudo boxes: {total_pseudo_boxes}")
    print(f"[Stats] Total merged boxes: {total_merged_boxes}")
    print(f"[Stats] Net change: {total_merged_boxes - total_gt_boxes} ({(total_merged_boxes - total_gt_boxes) / max(total_gt_boxes, 1) * 100:.1f}%)")
    print(f"[Stats] Images with new detections added: {n_merged_added}")
    print(f"\n[Result] Merged labels saved to: {MERGED_LABELS_DIR}")


if __name__ == "__main__":
    main()
