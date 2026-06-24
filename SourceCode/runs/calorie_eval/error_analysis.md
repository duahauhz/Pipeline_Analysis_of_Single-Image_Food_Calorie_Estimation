# Calorie evaluation — error analysis

- Images processed: **1733**
- Predictions: **1728**
- With GT (and prediction): **1728**
- Missing coin: **0**, no food: **5**, invalid coin bbox: **0**
- Total time: **73.4s** (23.6 img/s)

## Overall metrics

| metric | value |
|---|---|
| mae_g | 53.97 |
| rmse_g | 88.77 |
| bias_g | 1.49 |
| mae_kcal | 57.37 |
| rmse_kcal | 98.32 |
| mape_kcal | 0.4693 |
| accuracy_at_20pct | 0.3316 |

## Top-10 worst predictions (% kcal error)

| image_id | class | pred kcal | GT kcal | abs err kcal | abs % err |
|---|---|---|---|---|---|
| qiwi008 | qiwi | 387.3 | 31.8 | 355.5 | 1118.68% |
| bread005 | bread | 746.7 | 70.0 | 676.8 | 967.34% |
| qiwi008 | qiwi | 313.2 | 31.8 | 281.4 | 885.56% |
| bread007 | bread | 765.3 | 87.5 | 677.9 | 775.14% |
| qiwi008 | qiwi | 275.4 | 31.8 | 243.6 | 766.40% |
| qiwi008 | qiwi | 258.3 | 31.8 | 226.5 | 712.69% |
| bread004 | bread | 587.0 | 73.4 | 513.6 | 699.62% |
| bread005 | bread | 558.2 | 70.0 | 488.2 | 697.85% |
| bread006 | bread | 590.5 | 75.3 | 515.2 | 684.58% |
| bread006 | bread | 518.2 | 75.3 | 442.9 | 588.53% |

## Top-10 best predictions

| image_id | class | pred kcal | GT kcal | abs % err |
|---|---|---|---|---|
| peach004 | peach | 37.5 | 37.5 | 0.02% |
| lemon002 | lemon | 28.3 | 28.3 | 0.13% |
| mango007 | mango | 42.6 | 42.5 | 0.15% |
| plum004 | plum | 49.1 | 49.2 | 0.15% |
| egg007 | egg | 107.0 | 106.8 | 0.20% |
| mix008 | banana | 279.1 | 279.7 | 0.23% |
| plum004 | plum | 49.4 | 49.2 | 0.39% |
| egg004 | egg | 91.4 | 91.8 | 0.43% |
| litchi005 | litchi | 25.6 | 25.7 | 0.55% |
| sachima005 | sachima | 148.0 | 147.2 | 0.58% |

## Per-class MAE (g)

| class | n | MAE (g) | RMSE (g) | MAE (kcal) | MAPE |
|---|---|---|---|---|---|
| apple | 165 | 74.17 | 86.15 | 38.58 | 28.13% |
| banana | 102 | 127.78 | 147.65 | 110.41 | 79.86% |
| bread | 35 | 76.7 | 101.58 | 211.55 | 269.62% |
| bun | 58 | 38.97 | 43.53 | 86.89 | 50.03% |
| doughnut | 122 | 35.19 | 37.99 | 159.07 | 62.29% |
| egg | 60 | 13.07 | 17.72 | 18.72 | 19.85% |
| fired_dough_twist | 65 | 16.29 | 20.06 | 68.19 | 36.79% |
| grape | 34 | 295.41 | 366.19 | 203.83 | 134.58% |
| lemon | 149 | 32.77 | 48.85 | 14.81 | 26.05% |
| litchi | 48 | 6.22 | 7.31 | 4.1 | 14.39% |
| mango | 126 | 46.85 | 68.26 | 24.3 | 39.16% |
| mooncake | 68 | 37.78 | 49.31 | 161.79 | 55.91% |
| orange | 148 | 65.68 | 102.16 | 35.22 | 36.56% |
| peach | 78 | 32.82 | 56.4 | 16.18 | 31.85% |
| pear | 92 | 50.11 | 66.07 | 31.43 | 23.82% |
| plum | 94 | 25.12 | 33.11 | 11.37 | 23.47% |
| qiwi | 66 | 56.28 | 106.97 | 36.39 | 79.91% |
| sachima | 96 | 14.44 | 17.64 | 64.99 | 45.25% |
| tomato | 122 | 72.65 | 77.11 | 13.36 | 40.97% |
