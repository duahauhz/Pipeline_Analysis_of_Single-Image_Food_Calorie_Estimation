# Dual-view (top + side) calorie evaluation

## Mục đích

Tập dữ liệu ECUSTFD gốc chụp **mỗi món ăn từ 2 góc**:

- `*T(N).JPG` — top view (chụp từ trên xuống, dùng cho pipeline hiện tại)
- `*S(N).JPG` — side view (chụp từ bên hông)

Báo cáo trước đây chỉ dùng top view (1 ảnh) với depth ratio cố định nên MAPE
calo cao (~46.9%). Script này chạy **lại toàn bộ pipeline trên cùng cặp ảnh
T+S** để đo xem khi có cả side view thì MAPE cải thiện bao nhiêu.

## Cách chạy

### Bước 1 — Khôi phục ảnh ECUSTFD

Ảnh gốc cần được đặt lại vào `datasets/ECUSTFD/images/test/` với cấu trúc:

```
datasets/ECUSTFD/images/test/
  apple001T(1).JPG
  apple001S(1).JPG
  apple002T(1).JPG
  apple002S(1).JPG
  ...
```

Tên file phải giữ nguyên hậu tố `T(N)` / `S(N)` để script ghép cặp được.
ECUSTFD gốc có thể tải từ: https://github.com/llanmn3/ECUSTFDM

### Bước 2 — Chạy evaluation

Trên **Windows (PowerShell / cmd)**:

```powershell
scripts\run_dual_view_eval.bat
```

Trên **Linux / WSL**:

```bash
bash scripts/run_dual_view_eval.sh
```

Hoặc chạy trực tiếp:

```bash
python scripts/eval_calorie_dual_view.py \
    --source datasets/ECUSTFD/images/test \
    --weights runs/local_food_detect/output/weights/best.pt \
    --density-json data/density_processed.json \
    --output runs/dual_view_eval
```

### Bước 3 — Xem kết quả

Sau khi chạy xong, mở:

- `runs/dual_view_eval/comparison_1v_vs_2v.md` — bảng so sánh 1-ảnh vs 2-ảnh
- `runs/dual_view_eval/comparison_1v_vs_2v.csv` — bảng CSV
- `runs/dual_view_eval/per_image_predictions.csv` — chi tiết từng cặp ảnh
- `runs/dual_view_eval/per_class_metrics_v1.csv` / `..._v2.csv` — theo class
- `runs/dual_view_eval/error_analysis.md` — top-10 worst/best
- `runs/dual_view_eval/summary.json` — toàn bộ metrics dạng JSON

## Pipeline 2-view

```
T ảnh (top view)               S ảnh (side view)
       │                              │
   YOLOv13n                       YOLOv13n
       │                              │
 bbox food + coin              bbox food + coin
       │                              │
 mm/px_top                     mm/px_side
       │                              │
 length_mm = top bbox (lớn)     height_mm = side bbox (lớn)
 width_mm  = top bbox (nhỏ)     depth_mm  = side bbox (nhỏ)
       └──────────────┬──────────────────┘
                      ▼
              volume_cm3(L, W, D, geometry)
                      │
              mass = volume × density
                      │
              kcal  = mass × kcal_per_100g / 100
```

So với pipeline 1-view (cũ) dùng `depth = min(W, H) × depth_ratio` (0.55–0.90
tùy class), pipeline 2-view dùng **depth thực** từ silhouette của side view.

## Không cần train lại

YOLOv13n đã train xong từ experiment chính. Script này chỉ chạy **inference**
trên 2 ảnh thay vì 1, nên:

- Không cần GPU training
- Không cần tải lại dataset train
- Chỉ cần ảnh test gốc

Thời gian chạy ước tính: **5–15 phút** trên RTX 4050 (tuỳ số cặp ảnh).

## Tích hợp vào báo cáo

Sau khi chạy, copy số liệu từ `comparison_1v_vs_2v.md` vào bảng trong
`report/thesis.tex`. Đoạn văn mẫu nằm trong `THESIS_PATCH.tex` (cùng thư
mục `scripts/`).
