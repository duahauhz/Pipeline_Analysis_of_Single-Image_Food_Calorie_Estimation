# Calorie evaluation — error analysis

- Images processed: **1733**
- Predictions: **1724**
- With GT (and prediction): **1724**
- Missing coin: **0**, no food: **9**, invalid coin bbox: **0**
- Total time: **35.7s** (48.5 img/s)

## Overall metrics

| metric | value |
|---|---|
| mae_g | 55.23 |
| rmse_g | 96.53 |
| bias_g | 1.77 |
| mae_kcal | 57.17 |
| rmse_kcal | 99.1 |
| mape_kcal | 0.4522 |
| accuracy_at_20pct | 0.3306 |

## Top-10 worst predictions (% kcal error)

| image_id | class | pred kcal | GT kcal | abs err kcal | abs % err |
|---|---|---|---|---|---|
| bread005 | bread | 702.6 | 70.0 | 632.6 | 904.24% |
| bread006 | bread | 635.3 | 75.3 | 560.1 | 744.18% |
| bread004 | bread | 591.7 | 73.4 | 518.3 | 706.08% |
| bread005 | bread | 547.6 | 70.0 | 477.6 | 682.67% |
| bread006 | bread | 528.3 | 75.3 | 453.0 | 601.92% |
| pear005 | pear | 668.7 | 112.6 | 556.2 | 494.03% |
| bread007 | bread | 450.2 | 87.5 | 362.8 | 414.85% |
| bread007 | bread | 447.7 | 87.5 | 360.2 | 411.92% |
| grape002 | grape | 768.9 | 151.5 | 617.5 | 407.70% |
| bread007 | bread | 432.6 | 87.5 | 345.1 | 394.68% |

## Top-10 best predictions

| image_id | class | pred kcal | GT kcal | abs % err |
|---|---|---|---|---|
| mango006 | mango | 52.5 | 52.6 | 0.05% |
| sachima005 | sachima | 147.1 | 147.2 | 0.05% |
| lemon003 | lemon | 26.7 | 26.8 | 0.17% |
| apple016 | apple | 141.4 | 141.7 | 0.18% |
| sachima004 | sachima | 149.9 | 150.3 | 0.25% |
| plum003 | plum | 47.3 | 47.5 | 0.36% |
| lemon002 | lemon | 28.4 | 28.3 | 0.40% |
| sachima004 | sachima | 151.0 | 150.3 | 0.49% |
| litchi003 | litchi | 33.8 | 34.0 | 0.50% |
| mix008 | banana | 281.1 | 279.7 | 0.50% |

## Per-class MAE (g)

| class | n | MAE (g) | RMSE (g) | MAE (kcal) | MAPE |
|---|---|---|---|---|---|
| apple | 165 | 80.56 | 106.14 | 43.94 | 32.16% |
| banana | 102 | 136.97 | 148.98 | 118.82 | 85.32% |
| bread | 34 | 79.58 | 102.67 | 209.31 | 267.57% |
| bun | 58 | 38.4 | 43.14 | 86.57 | 49.77% |
| doughnut | 122 | 35.66 | 38.31 | 161.18 | 63.20% |
| egg | 60 | 12.96 | 17.7 | 15.7 | 16.47% |
| fired_dough_twist | 66 | 15.04 | 17.39 | 65.38 | 35.22% |
| grape | 34 | 352.05 | 436.9 | 242.91 | 160.39% |
| lemon | 149 | 33.47 | 48.65 | 15.7 | 28.30% |
| litchi | 48 | 5.96 | 7.28 | 3.93 | 13.65% |
| mango | 125 | 37.59 | 51.26 | 20.99 | 34.06% |
| mooncake | 68 | 33.38 | 45.84 | 140.22 | 50.31% |
| orange | 147 | 65.22 | 95.56 | 32.9 | 33.95% |
| peach | 78 | 26.84 | 39.55 | 14.23 | 29.17% |
| pear | 93 | 66.48 | 127.06 | 37.16 | 29.06% |
| plum | 94 | 20.25 | 23.15 | 9.31 | 19.26% |
| qiwi | 63 | 32.72 | 38.27 | 19.92 | 27.76% |
| sachima | 96 | 13.71 | 17.23 | 61.71 | 43.00% |
| tomato | 122 | 78.25 | 81.26 | 14.12 | 43.32% |
