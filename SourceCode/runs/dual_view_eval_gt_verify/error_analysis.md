# Dual-view (GT bboxes) — error analysis

- Pairs evaluated: **158**
- v2 valid: **158**

## Top-10 worst (by 2-view % kcal error)

| pair_id | class | GT kcal | v1 %err | v2 %err | Δ |
|---|---|---:|---:|---:|---:|
| bread006 | bread | 75.3 | 406.83% | 406.83% | +0.00 pp |
| bread002 | bread | 76.3 | 296.64% | 336.21% | +39.57 pp |
| bread003 | bread | 68.9 | 567.44% | 317.15% | -250.29 pp |
| bread005 | bread | 70.0 | 514.41% | 311.15% | -203.26 pp |
| grape001 | grape | 309.1 | 255.37% | 240.07% | -15.31 pp |
| grape002 | grape | 151.5 | 199.96% | 233.48% | +33.52 pp |
| bread001 | bread | 70.5 | 229.69% | 181.04% | -48.65 pp |
| banana004 | banana | 142.0 | 190.60% | 162.79% | -27.81 pp |
| bread007 | bread | 87.5 | 172.77% | 149.06% | -23.71 pp |
| banana002 | banana | 139.6 | 220.53% | 146.03% | -74.50 pp |

## Top-10 best

| pair_id | class | GT kcal | v1 %err | v2 %err | Δ |
|---|---|---:|---:|---:|---:|
| bun001 | bun | 179.3 | 55.25% | 0.56% | -54.69 pp |
| litchi001 | litchi | 25.3 | 4.60% | 0.63% | -3.97 pp |
| apple005 | apple | 110.5 | 0.61% | 0.71% | +0.10 pp |
| lemon003 | lemon | 26.8 | 7.86% | 1.01% | -6.85 pp |
| plum001 | plum | 69.8 | 5.97% | 1.02% | -4.96 pp |
| peach005 | peach | 61.8 | 3.56% | 1.35% | -2.21 pp |
| apple007 | apple | 169.0 | 3.11% | 1.39% | -1.72 pp |
| orange007 | orange | 86.9 | 12.00% | 1.47% | -10.53 pp |
| tomato003 | tomato | 33.6 | 6.21% | 1.78% | -4.42 pp |
| egg003 | egg | 84.5 | 19.94% | 2.03% | -17.91 pp |
