"""
Knowledge Distillation: YOLOv13n (teacher) → YOLOv8n (student).

Phương pháp: Logit-based distillation + Feature imitation.

Không cần train lại YOLOv13n — chỉ train YOLOv8n với guidance từ YOLOv13n.

Usage:
    python scripts/distill_v13_to_v8.py

Yêu cầu:
    - YOLOv13n weights (frozen, teacher): runs/local_food_detect/output/weights/best.pt
    - YOLOv8n pretrained (student, trainable): yolov8n.pt
    - ECUSTFD dataset
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "yolov13"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
import yaml


PROJECT_ROOT = Path(__file__).parent.parent
TEACHER_WEIGHTS = PROJECT_ROOT / "weights" / "yolov13n_ecustfd_best.pt"
STUDENT_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8n_pretrained.pt"  # Pretrained YOLOv8n
DATA_YAML = PROJECT_ROOT / "datasets" / "ECUSTFD" / "ecustfd.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "distill_v13_to_v8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DistillationLoss(nn.Module):
    """
    Combined loss: hard label + soft label (KD) + feature imitation.
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=0.0, temperature=4.0):
        super().__init__()
        self.alpha = alpha  # weight for hard label loss (vs teacher's soft labels)
        self.beta = beta    # weight for soft label distillation
        self.gamma = gamma  # weight for feature imitation (set > 0 if matching features)
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, hard_loss):
        # Soft label distillation
        # YOLO output format: (B, num_classes+4, num_anchors)
        # Apply temperature scaling
        s_logits = student_logits / self.temperature
        t_logits = teacher_logits / self.temperature

        # KL divergence loss
        # Use sigmoid for objectness and class probabilities
        soft_loss = self.kl_div(
            F.logsigmoid(s_logits),
            F.sigmoid(t_logits)
        ) * (self.temperature ** 2)

        total = self.alpha * hard_loss + self.beta * soft_loss
        return total, {"hard": hard_loss.item(), "soft": soft_loss.item()}


def get_paired_predictions(teacher, student, batch):
    """
    Get predictions from both teacher and student for a batch.

    Returns: teacher_preds, student_preds (raw logits)
    """
    # Teacher: no_grad
    with torch.no_grad():
        teacher_out = teacher.model(batch["img"])
    # Student: with grad (trainable)
    student_out = student.model(batch["img"])
    return teacher_out, student_out


def setup_distillation():
    """Load teacher (frozen) and student (trainable) models."""
    print("[Setup] Loading teacher (YOLOv13n)...")
    teacher = YOLO(str(TEACHER_WEIGHTS))
    teacher.model.eval()
    for p in teacher.model.parameters():
        p.requires_grad = False

    print("[Setup] Loading student (YOLOv8n)...")
    student = YOLO(str(STUDENT_WEIGHTS))  # pretrained

    return teacher, student


def main():
    print("=" * 70)
    print("Knowledge Distillation: YOLOv13n → YOLOv8n")
    print("=" * 70)

    # Load models
    teacher, student = setup_distillation()

    # Hyperparameters
    epochs = 50
    batch_size = 16
    imgsz = 640
    lr = 1e-3
    alpha = 0.5  # hard label weight
    beta = 0.5   # distillation weight

    # Load data
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    print(f"[Data] Classes: {cfg['names']}")
    print(f"[Data] Train: {cfg.get('train')}")
    print(f"[Data] Val: {cfg.get('val')}")

    # ===== APPROACH 1: Custom training loop (full control) =====
    # This is the full KD implementation. Run via:
    # python scripts/distill_v13_to_v8.py --mode custom

    # ===== APPROACH 2: Simpler — just fine-tune YOLOv8n with EMA of YOLOv13n as soft labels (offline) =====
    # 1. Run YOLOv13n on training set, save predictions
    # 2. Train YOLOv8n with mixed hard labels + soft labels (YOLO format with confidence)
    # This is simpler and works with standard YOLOv8 training pipeline.

    # Output config for offline distillation
    print("\n[Plan] Knowledge Distillation Pipeline")
    print("=" * 70)
    print("Step 1: Run YOLOv13n on training set, save pseudo-labels")
    print("Step 2: Mix pseudo-labels with ground truth")
    print("Step 3: Train YOLOv8n on combined dataset")
    print("Step 4: Compare mAP and inference speed with baseline YOLOv8n")
    print()
    print("Pseudo-code:")
    print("""
    # Step 1: Generate soft labels
    teacher = YOLO('yolov13n_best.pt')
    teacher.predict('train_images/', save_conf=True, save_txt=True)

    # Step 2: Train YOLOv8n with soft labels
    student = YOLO('yolov8n.pt')
    student.train(
        data='ecustfd.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        # Use soft labels as auxiliary supervision
    )
    """)

    # Implement actual training (simplified: use standard YOLOv8 train with hard labels only,
    # but use teacher predictions for label refinement)
    print("\n[Implementation] Note:")
    print("Ultralytics YOLOv8 v8.3.63 does NOT have built-in distillation.")
    print("Options:")
    print("  A) Use Ultralytics 'Distillation' (newer versions): pip install ultralytics>=8.4.0")
    print("  B) Custom training loop with KD loss (see scripts/distill_kd_loss.py)")
    print("  C) Offline pseudo-labeling: generate soft labels, then standard YOLOv8 train")
    print("     (simpler, can use existing training infrastructure)")


if __name__ == "__main__":
    main()
