# Statistical analysis of the calorie pipeline

- Baseline file: `runs\calorie_eval\per_image_predictions.csv`
- Comparison file: `runs\calorie_eval_yolov8n\per_image_predictions.csv`
- Bootstrap resamples: 5000

## 1. Baseline bootstrap 95% CIs

| metric | point | CI low | CI high |
|---|---|---|---|
| mae_kcal | 57.3708 | 53.6937 | 61.2089 |
| mae_g | 53.9718 | 50.6654 | 57.3157 |
| mape_kcal | 0.4693 | 0.4364 | 0.5063 |

## 2. Per-class bootstrap 95% CIs (baseline)

| class | n | MAE (g) | MAE CI | MAPE | MAPE CI |
|---|---|---|---|---|---|
| apple | 165 | 74.17 | [67.54, 80.92] | 28.13% | [25.55%, 30.68%] |
| banana | 102 | 127.78 | [114.95, 143.64] | 79.86% | [73.22%, 87.44%] |
| bread | 35 | 76.70 | [55.35, 99.17] | 269.62% | [188.75%, 354.39%] |
| bun | 58 | 38.97 | [33.69, 43.86] | 50.03% | [43.55%, 56.01%] |
| doughnut | 122 | 35.19 | [32.61, 37.69] | 62.29% | [58.00%, 66.48%] |
| egg | 60 | 13.07 | [10.21, 16.30] | 19.85% | [15.52%, 24.54%] |
| fired_dough_twist | 65 | 16.29 | [13.64, 19.34] | 36.79% | [31.92%, 41.77%] |
| grape | 34 | 295.41 | [221.17, 368.22] | 134.58% | [100.76%, 167.75%] |
| lemon | 149 | 32.77 | [27.00, 38.93] | 26.05% | [22.00%, 30.43%] |
| litchi | 48 | 6.22 | [5.16, 7.32] | 14.39% | [11.84%, 17.10%] |
| mango | 126 | 46.85 | [38.56, 55.95] | 39.16% | [34.61%, 43.76%] |
| orange | 148 | 65.68 | [54.07, 78.97] | 36.56% | [27.89%, 47.32%] |
| mooncake | 68 | 37.78 | [30.36, 45.50] | 55.91% | [49.18%, 62.39%] |
| peach | 78 | 32.82 | [23.60, 43.60] | 31.85% | [23.79%, 41.20%] |
| pear | 92 | 50.11 | [41.63, 59.35] | 23.82% | [18.16%, 30.51%] |
| plum | 94 | 25.12 | [21.17, 29.62] | 23.47% | [20.00%, 27.23%] |
| qiwi | 66 | 56.28 | [36.94, 81.58] | 79.91% | [37.98%, 136.62%] |
| sachima | 96 | 14.44 | [12.41, 16.45] | 45.25% | [38.85%, 51.46%] |
| tomato | 122 | 72.65 | [68.20, 77.28] | 40.97% | [38.21%, 43.98%] |

## 3. Paired Wilcoxon test: baseline vs comparison

- n common: 1719
- mean abs err (baseline): 57.5315
- mean abs err (comparison): 57.0115
- mean paired diff (baseline - comparison): 0.5200
- Wilcoxon statistic: 697034.0000
- p-value: 4.0648e-02
- rank-biserial effect size: 0.0506

A positive `mean_diff` means baseline error > comparison error (i.e. baseline is worse on calorie MAE). A positive `rank_biserial` indicates the comparison has more pairs with smaller error.

## 4. Ablation paired Wilcoxon: per_class vs geometry_fallback

- n common: 1728
- mean abs err (per_class): 57.3708
- mean abs err (geometry_fallback): 63.9311
- p-value: 1.1732e-10
- rank-biserial effect size: -0.0891

## 5. Ablation paired Wilcoxon: per_image vs per_class

- n common: 1728
- mean abs err (per_image): 12.3400
- mean abs err (per_class): 57.3708
- p-value: 1.2176e-194
- rank-biserial effect size: -0.8634
