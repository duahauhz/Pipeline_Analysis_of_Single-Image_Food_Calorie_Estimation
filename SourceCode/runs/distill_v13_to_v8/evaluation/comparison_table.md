# So sánh model trên ECUSTFD test split

Môi trường: RTX 4050 Laptop GPU, 6GB VRAM, batch=16, imgsz=640, workers=0.

Test split: 100 ảnh lấy mẫu (lấy ngẫu nhiên từ thư mục `datasets/ECUSTFD/images/test`).


| Model | Params (M) | mAP50 | mAP50-95 | Precision | Recall | Latency (ms) | FPS | Device |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| yolov8n_baseline | 3.01 | 0.9908 | 0.9093 | 0.9898 | 0.9930 | 19.47 | 51.4 | cuda:0 |
| yolov13n_local | 2.45 | 0.9923 | 0.9173 | 0.9943 | 0.9972 | 34.34 | 29.1 | cuda:0 |
| yolov8n_local | 3.01 | 0.9924 | 0.9162 | 0.9892 | 0.9978 | 18.27 | 54.7 | cuda:0 |
| yolov13_to_v8n_distilled | 3.01 | 0.9913 | 0.9124 | 0.9867 | 0.9863 | 18.93 | 52.8 | cuda:0 |

## Nhận xét nhanh

- `yolov8n_baseline`: train local gốc từ scratch trên ECUSTFD (baseline YOLOv8n)
- `yolov13n_local`: train YOLOv13n gốc trên ECUSTFD (so sánh kiến trúc)
- `yolov8n_local`: train YOLOv8n trên ECUSTFD (so sánh dataset/config khác)
- `yolov13_to_v8n_distilled`: student YOLOv8n được distill từ teacher YOLOv13n (kiểm tra knowledge transfer)