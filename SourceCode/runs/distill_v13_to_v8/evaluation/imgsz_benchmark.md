# YOLOv13n_local — imgsz tradeoff benchmark

- Model: `yolov13n_ecustfd_local/best.pt`
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)
- Test images: 50 từ `datasets/ECUSTFD/images/test`

| imgsz | mAP50 | mAP50-95 | Latency (ms) | FPS | Δ mAP50-95 vs 640 | Δ FPS vs 640 |
|---:|---:|---:|---:|---:|---:|---:|
| 320 | 0.9923 | 0.9006 | 85.13 | 11.7 | +1.67 pp | 1.55x |
| 416 | 0.9923 | 0.9081 | 80.60 | 12.4 | +0.92 pp | 1.63x |
| 512 | 0.9919 | 0.9139 | 95.50 | 10.5 | +0.35 pp | 1.38x |
| 640 | 0.9923 | 0.9173 | 131.60 | 7.6 | +0.00 pp | 1.00x |

## Kết luận
- **Sweet spot (giảm ≤ 1 pp mAP50-95, tăng FPS)**: imgsz=416 (12.4 FPS, mAP drop 0.92 pp) hoặc imgsz=512 (10.5 FPS, mAP drop 0.35 pp).
- Nếu paper cần so sánh với YOLOv8n (51-55 FPS): **imgsz=416 là tốt nhất** — chỉ mất ~1 pp mAP nhưng nhanh hơn 1.6x.
- Nếu paper nhấn mạnh accuracy tối đa: giữ imgsz=640 (mAP50-95 = 0.9173).

### Ghi chú quan trọng về latency benchmark

- Latency đo ở đây bao gồm **toàn bộ `model.predict()` pipeline** (load image + preprocess resize từ 3000x4000 → imgsz + GPU transfer + inference + NMS + transfer về CPU).
- Inference thuần (GPU) chỉ chiếm **8.1 ms @ imgsz=640** → tốc độ thực của model là ~125 FPS.
- Preprocess là bottleneck: ảnh gốc lớn, resize trên CPU trước khi đưa vào GPU.
- Trong deployment thực tế (ảnh đầu vào đã ở 640x640 từ camera/mobile), FPS sẽ cao hơn nhiều so với benchmark này.