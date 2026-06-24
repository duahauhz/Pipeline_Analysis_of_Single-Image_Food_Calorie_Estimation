# Calorie pipeline ablation

- Weights: `runs\local_food_detect\output\weights\best.pt`
- Source: `datasets\ECUSTFD\images\test`
- Images: 1733
- Density source: `data\density_processed.json`

## Overall metrics

| policy | n | MAE (g) | RMSE (g) | bias (g) | MAE (kcal) | MAPE | acc@20% |
|---|---|---|---|---|---|---|---|
| per_image | 1728 | 17.34 | 72.65 | 17.34 | 12.34 | 10.35% | 89.93% |
| per_class | 1728 | 53.97 | 88.77 | 1.49 | 57.37 | 46.93% | 33.16% |
| geometry_fallback | 1728 | 55.81 | 90.15 | -0.21 | 63.93 | 52.25% | 29.75% |

## Per-class MAE (g)

| class | n | per_image | per_class | geometry_fallback |
|---|---|---|---|---|
| apple | 165 | 1.55 | 165 | 74.17 | 165 | 78.43 |
| banana | 102 | 37.07 | 102 | 127.78 | 102 | 131.04 |
| bread | 35 | 3.32 | 35 | 76.7 | 35 | 108.98 |
| bun | 58 | 1.19 | 58 | 38.97 | 58 | 38.32 |
| doughnut | 122 | 0.54 | 122 | 35.19 | 122 | 37.48 |
| egg | 60 | 5.91 | 60 | 13.07 | 60 | 14.41 |
| fired_dough_twist | 65 | 0.59 | 65 | 16.29 | 65 | 26.86 |
| grape | 34 | 0.0 | 34 | 295.41 | 34 | 298.07 |
| lemon | 149 | 78.05 | 149 | 32.77 | 149 | 32.22 |
| litchi | 48 | 0.0 | 48 | 6.22 | 48 | 6.31 |
| mango | 126 | 35.33 | 126 | 46.85 | 126 | 45.88 |
| mooncake | 68 | 17.84 | 68 | 37.78 | 68 | 36.16 |
| orange | 148 | 35.43 | 148 | 65.68 | 148 | 61.65 |
| peach | 78 | 14.6 | 78 | 32.82 | 78 | 30.87 |
| pear | 92 | 10.8 | 92 | 50.11 | 92 | 52.64 |
| plum | 94 | 2.27 | 94 | 25.12 | 94 | 26.22 |
| qiwi | 66 | 3.16 | 66 | 56.28 | 66 | 55.12 |
| sachima | 96 | 0.0 | 96 | 14.44 | 96 | 25.01 |
| tomato | 122 | 1.53 | 122 | 72.65 | 122 | 70.19 |
