"""
Generate figures for the paper.
Creates: pipeline diagram, per-class MAPE chart, error distribution, batch speedup, KD comparison.
"""
import sys
from pathlib import Path
import json

# Setup path for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_pipeline_diagram():
    """Create pipeline architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title('Figure 1: End-to-End Calorie Estimation Pipeline', fontsize=14, fontweight='bold', pad=20)

    # Define boxes
    boxes = [
        (0.5, 1.3, 'Input\nImage', '#E3F2FD'),
        (2.5, 1.3, 'YOLOv13n\nDetector', '#BBDEFB'),
        (4.5, 1.3, 'Bounding\nBoxes', '#E1F5FE'),
        (6.5, 1.3, 'Scale\nCalibration\n(coin 25mm)', '#B3E5FC'),
        (8.5, 1.3, 'Geometric\nVolume\nModel', '#81D4FA'),
        (10.5, 1.3, 'Mass\n(vol × density)', '#4FC3F7'),
    ]

    # Draw boxes
    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), 1.5, 1.2,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='#1976D2', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.75, y + 0.6, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.5
        x2 = boxes[i + 1][0]
        y = 1.9
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))

    # Add calorie output
    ax.annotate('', xy=(11.3, 1.0), xytext=(11.0, 1.0),
               arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2))
    ax.text(11.8, 1.0, 'Calorie\n(kcal)', ha='left', va='center', fontsize=9,
            fontweight='bold', color='#D32F2F')

    # Add labels
    ax.text(1.25, 2.7, 'Stage 1', ha='center', fontsize=8, color='#666')
    ax.text(3.25, 2.7, 'Stage 2', ha='center', fontsize=8, color='#666')
    ax.text(5.25, 2.7, 'Stage 3', ha='center', fontsize=8, color='#666')
    ax.text(7.25, 2.7, 'Stage 4', ha='center', fontsize=8, color='#666')
    ax.text(9.25, 2.7, 'Stage 5', ha='center', fontsize=8, color='#666')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_pipeline_diagram.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / "fig1_pipeline_diagram.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] fig1_pipeline_diagram.png/pdf")
    plt.close()


def create_per_class_mape_chart():
    """Create per-class MAPE bar chart."""
    data = [
        ('litchi', 14.39, 48),
        ('egg', 19.85, 60),
        ('plum', 23.47, 94),
        ('pear', 23.82, 92),
        ('lemon', 26.05, 149),
        ('apple', 28.13, 165),
        ('peach', 31.85, 78),
        ('orange', 36.56, 148),
        ('fired_dough_twist', 36.79, 65),
        ('mango', 39.16, 126),
        ('tomato', 40.97, 122),
        ('sachima', 45.25, 96),
        ('bun', 50.03, 58),
        ('mooncake', 55.91, 68),
        ('doughnut', 62.29, 122),
        ('banana', 79.86, 102),
        ('qiwi', 79.91, 66),
        ('grape', 134.58, 34),
        ('bread', 269.62, 35),
    ]

    classes = [d[0] for d in data]
    mapes = [d[1] for d in data]
    counts = [d[2] for d in data]

    # Color based on MAPE
    colors = []
    for m in mapes:
        if m < 30:
            colors.append('#4CAF50')
        elif m < 50:
            colors.append('#FFC107')
        elif m < 100:
            colors.append('#FF9800')
        else:
            colors.append('#F44336')

    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(range(len(classes)), mapes, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, m, c in zip(bars, mapes, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{m:.1f}%\n(n={c})',
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('MAPE (%)', fontsize=11)
    ax.set_xlabel('Food Class', fontsize=11)
    ax.set_title('Figure 2: Per-Class MAPE on Calorie Estimation\n(Sorted by Error)', fontsize=13, fontweight='bold')

    # Add horizontal lines
    ax.axhline(y=25, color='#4CAF50', linestyle='--', alpha=0.7, label='Good (<25%)')
    ax.axhline(y=50, color='#FF9800', linestyle='--', alpha=0.7, label='Moderate (<50%)')

    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 320)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_per_class_mape.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(OUTPUT_DIR / "fig2_per_class_mape.pdf", bbox_inches='tight',
                facecolor='white')
    print(f"[OK] fig2_per_class_mape.png/pdf")
    plt.close()


def create_error_distribution():
    """Create error distribution histogram."""
    # Load prediction data
    pred_file = PROJECT_ROOT / "runs" / "calorie_eval" / "per_image_predictions.csv"
    if not pred_file.exists():
        print(f"[SKIP] fig3_error_distribution: {pred_file} not found")
        return

    try:
        import pandas as pd
        df = pd.read_csv(pred_file)

        df['pct_error'] = (df['predicted_kcal'] - df['gt_kcal']).abs() / df['gt_kcal'] * 100
        pct_errors = df['pct_error']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1 = axes[0]
        ax1.hist(pct_errors, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
        ax1.axvline(x=20, color='#F44336', linestyle='--', linewidth=2, label='Accuracy@20% threshold')
        ax1.set_xlabel('Percentage Error (%)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Figure 3a: Distribution of Percentage Errors', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Box plot by class (top 10 worst mean error)
        ax2 = axes[1]
        top10 = df.groupby('class')['pct_error'].mean().nlargest(10).index.tolist()

        data_to_plot = []
        labels = []
        for cls in top10[:8]:
            cls_errors = df[df['class'] == cls]
            cls_pct = cls_errors['pct_error']
            data_to_plot.append(cls_pct.values)
            labels.append(cls)

        bp = ax2.boxplot(data_to_plot, patch_artist=True)
        colors_bp = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(data_to_plot)))
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Percentage Error (%)', fontsize=11)
        ax2.set_title('Figure 3b: Error Distribution by Worst Classes', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig3_error_distribution.png", dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.savefig(OUTPUT_DIR / "fig3_error_distribution.pdf", bbox_inches='tight',
                    facecolor='white')
        print(f"[OK] fig3_error_distribution.png/pdf")
        plt.close()
    except Exception as e:
        print(f"[SKIP] fig3_error_distribution: {e}")


def create_batch_speedup_chart():
    """Create batch processing speedup chart."""
    data = [
        ('batch=1', 34.54, 29.0),
        ('batch=2', 25.74, 38.9),
        ('batch=4', 19.61, 51.0),
        ('batch=8', 15.53, 64.4),
        ('batch=16', 15.09, 66.3),
    ]

    batches = [d[0] for d in data]
    latencies = [d[1] for d in data]
    fps = [d[2] for d in data]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for latency
    x = np.arange(len(batches))
    bars = ax1.bar(x - 0.2, latencies, 0.4, label='Latency (ms)', color='#2196F3', alpha=0.8)

    ax1.set_xlabel('Batch Size', fontsize=11)
    ax1.set_ylabel('Latency (ms per image)', fontsize=11, color='#2196F3')
    ax1.tick_params(axis='y', labelcolor='#2196F3')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batches)

    # Add value labels
    for bar, lat, f in zip(bars, latencies, fps):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{lat:.1f}ms\n{f:.0f} fps', ha='center', va='bottom', fontsize=9)

    # Line chart for speedup
    ax2 = ax1.twinx()
    speedups = [34.54 / lat for lat in latencies]
    ax2.plot(x, speedups, 'o-', color='#4CAF50', linewidth=2, markersize=8, label='Speedup')
    ax2.set_ylabel('Speedup (× vs batch=1)', fontsize=11, color='#4CAF50')
    ax2.tick_params(axis='y', labelcolor='#4CAF50')

    ax1.set_title('Figure 4: Batch Processing Speedup on YOLOv13n\n(RTX 4050)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_batch_speedup.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(OUTPUT_DIR / "fig4_batch_speedup.pdf", bbox_inches='tight',
                facecolor='white')
    print(f"[OK] fig4_batch_speedup.png/pdf")
    plt.close()


def create_kd_comparison_chart():
    """Create knowledge distillation comparison chart."""
    models = ['YOLOv13n\n(Teacher)', 'YOLOv8n\n(Baseline)', 'YOLOv8n\n+KD']
    map50 = [99.23, 99.08, 99.21]
    map50_95 = [91.37, 90.82, 91.09]
    params = [2.46, 3.01, 3.01]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # mAP comparison
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax1.bar(x - width/2, map50, width, label='mAP@0.5', color='#2196F3')
    bars2 = ax1.bar(x + width/2, map50_95, width, label='mAP@0.5:0.95', color='#4CAF50')
    ax1.set_ylabel('mAP (%)', fontsize=11)
    ax1.set_title('Detection Performance\n(Validation Set)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(88, 100)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    # Model size comparison
    ax2 = axes[1]
    colors = ['#FF9800', '#9E9E9E', '#9C27B0']
    bars = ax2.bar(models, params, color=colors, alpha=0.8)
    ax2.set_ylabel('Parameters (M)', fontsize=11)
    ax2.set_title('Model Size Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, p in zip(bars, params):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{p:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Improvement delta
    ax3 = axes[2]
    deltas = [0, -0.15, +0.13]
    colors_delta = ['#FF9800', '#9E9E9E', '#4CAF50']
    bars = ax3.bar(models, deltas, color=colors_delta, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('ΔmAP@0.5 vs Baseline (%)', fontsize=11)
    ax3.set_title('KD Improvement over Baseline', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, d in zip(bars, deltas):
        y_pos = bar.get_height() + 0.02 if bar.get_height() >= 0 else bar.get_height() - 0.08
        sign = '+' if d > 0 else ''
        ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{sign}{d:.2f}', ha='center', va='bottom' if d >= 0 else 'top',
                fontsize=10, fontweight='bold')

    plt.suptitle('Knowledge Distillation: YOLOv13n (Teacher) → YOLOv8n (Student)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_kd_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(OUTPUT_DIR / "fig5_kd_comparison.pdf", bbox_inches='tight',
                facecolor='white')
    print(f"[OK] fig5_kd_comparison.png/pdf")
    plt.close()


def create_model_comparison_chart():
    """Create YOLOv13n vs YOLOv8n comparison chart."""
    metrics = ['mAP@0.5\n×100', 'mAP@0.5:0.95\n×100', 'Precision\n×100', 'Recall\n×100']
    v13 = [99.23, 91.73, 99.45, 99.73]
    v8 = [99.08, 90.93, 98.90, 99.26]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, v13, width, label='YOLOv13n', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, v8, width, label='YOLOv8n', color='#FF9800', alpha=0.8)

    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Figure 6: Detection Performance Comparison\n(Validation Set)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(88, 102)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    # Add difference annotations
    for i, (v1, v2) in enumerate(zip(v13, v8)):
        diff = v1 - v2
        if diff > 0:
            ax.annotate(f'+{diff:.2f}', xy=(i, max(v1, v2) + 1.5),
                       ha='center', fontsize=8, color='#4CAF50', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_model_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(OUTPUT_DIR / "fig6_model_comparison.pdf", bbox_inches='tight',
                facecolor='white')
    print(f"[OK] fig6_model_comparison.png/pdf")
    plt.close()


def create_ablation_chart():
    """Create density ablation study chart."""
    policies = ['per_image\n(GT)', 'per_class\n(Ours)', 'geometry\n_fallback']
    maes = [17.34, 53.97, 55.81]
    mapes = [10.35, 46.93, 52.25]
    acc20 = [89.93, 33.16, 29.75]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(policies))
    width = 0.25

    # MAE bars
    bars1 = ax1.bar(x - width, maes, width, label='MAE (g)', color='#F44336', alpha=0.8)
    ax1.set_ylabel('MAE (g)', fontsize=11, color='#F44336')
    ax1.tick_params(axis='y', labelcolor='#F44336')
    ax1.set_ylim(0, 70)

    # MAPE bars (secondary axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, mapes, width, label='MAPE (%)', color='#FF9800', alpha=0.8)
    ax2.set_ylabel('MAPE (%)', fontsize=11, color='#FF9800')
    ax2.tick_params(axis='y', labelcolor='#FF9800')
    ax2.set_ylim(0, 70)

    # Accuracy bars (tertiary)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    bars3 = ax3.bar(x + width, acc20, width, label='Acc@20%', color='#4CAF50', alpha=0.8)
    ax3.set_ylabel('Accuracy@20% (%)', fontsize=11, color='#4CAF50')
    ax3.tick_params(axis='y', labelcolor='#4CAF50')
    ax3.set_ylim(0, 100)

    ax1.set_xlabel('Density Policy', fontsize=11)
    ax1.set_title('Figure 7: Density Source Ablation Study\n(YOLOv13n on Test Set, n=1728)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')

    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_ablation.png", dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig(OUTPUT_DIR / "fig7_ablation.pdf", bbox_inches='tight',
                facecolor='white')
    print(f"[OK] fig7_ablation.png/pdf")
    plt.close()


def main():
    print("=" * 50)
    print("Generating Paper Figures")
    print("=" * 50)

    create_pipeline_diagram()
    create_per_class_mape_chart()
    create_error_distribution()
    create_batch_speedup_chart()
    create_kd_comparison_chart()
    create_model_comparison_chart()
    create_ablation_chart()

    print("\n" + "=" * 50)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
