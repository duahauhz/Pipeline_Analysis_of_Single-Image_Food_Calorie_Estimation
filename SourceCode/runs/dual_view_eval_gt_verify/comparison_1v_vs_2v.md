# So sánh 1-ảnh (top) vs 2-ảnh (top + side) trên ECUSTFD

- Số cặp ảnh T+S có cả label top & side: **158**
- Detector: **ground-truth bbox** (loại trừ nhiễu detection để cô lập đóng góp của geometry)
- Density source: `data/density_processed.json` (per-class)
- Thời gian: **0.1s**

## Overall metrics

| Setup | n | MAPE (kcal) | MAE (g) | Acc@20% | Ghi chú |
|---|---:|---:|---:|---:|---|
| 1 view (top only) | 158 | 48.40% | 61.59 | 33.54% | depth = min(W,H) × depth_ratio (per class) |
| 2 views (top + side) | 158 | 46.57% | 58.46 | 42.41% | real L×W×D từ 2 silhouette |
| ECUSTFD paper baseline (Liang & Li 2017) | 297 | 18.9% | — | — | Dual-camera + GrabCut |

## Per-class MAPE

| class | n (1v) | MAPE 1v | n (2v) | MAPE 2v | Δ (2v − 1v) |
|---|---:|---:|---:|---:|---:|
| apple | 23 | 25.11% | 23 | 24.24% | -0.87 pp |
| banana | 18 | 74.90% | 18 | 79.09% | +4.19 pp |
| bread | 7 | 343.75% | 7 | 256.55% | -87.20 pp |
| bun | 8 | 51.72% | 8 | 19.09% | -32.63 pp |
| doughnut | 9 | 9.64% | 9 | 69.55% | +59.91 pp |
| egg | 6 | 25.89% | 6 | 7.61% | -18.28 pp |
| fired_dough_twist | 7 | 36.42% | 7 | 66.29% | +29.87 pp |
| grape | 2 | 227.67% | 2 | 236.77% | +9.10 pp |
| lemon | 8 | 29.18% | 8 | 21.65% | -7.53 pp |
| litchi | 5 | 15.64% | 5 | 9.16% | -6.48 pp |
| mango | 11 | 32.31% | 11 | 14.24% | -18.07 pp |
| mooncake | 6 | 34.07% | 6 | 54.71% | +20.64 pp |
| orange | 16 | 22.88% | 16 | 22.25% | -0.63 pp |
| peach | 5 | 18.97% | 5 | 13.23% | -5.74 pp |
| pear | 6 | 29.72% | 6 | 20.47% | -9.25 pp |
| plum | 4 | 15.39% | 4 | 9.31% | -6.08 pp |
| qiwi | 8 | 25.89% | 8 | 19.34% | -6.55 pp |
| sachima | 5 | 23.30% | 5 | 68.51% | +45.21 pp |
| tomato | 4 | 13.37% | 4 | 8.78% | -4.59 pp |

## Nhận xét
- 2-view pipeline cải thiện MAPE -1.83 pp so với 1-view (48.40% → 46.57%).
- Con số này là **upper bound** (detector hoàn hảo). Khi chạy với YOLOv13n inference thật, MAPE sẽ tệ hơn do sai số detection, nhưng delta giữa 1-view và 2-view vẫn phản ánh đúng đóng góp của geometry.
