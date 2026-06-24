# Per-class metrics — YOLOv13n_local trên val split (623 ảnh)

- **mAP50** (toàn bộ): 0.9923
- **mAP50-95** (toàn bộ): 0.9173
- **Precision** (toàn bộ): 0.9943
- **Recall** (toàn bộ): 0.9972
- **Device**: cuda:0

## Bảng per-class (sắp xếp theo mAP50-95 giảm dần)

| # | Class | mAP50 | mAP50-95 | Precision | Recall | F1 | Nhận xét |
|--:|---|---:|---:|---:|---:|---:|---|
| 1 | apple | 0.9950 | 0.9690 | 0.9992 | 1.0000 | 0.9996 | Xuất sắc |
| 2 | doughnut | 0.9950 | 0.9591 | 0.9984 | 1.0000 | 0.9992 | Xuất sắc |
| 3 | mango | 0.9950 | 0.9575 | 0.9986 | 1.0000 | 0.9993 | Xuất sắc |
| 4 | orange | 0.9950 | 0.9491 | 0.9985 | 1.0000 | 0.9992 | Tốt |
| 5 | grape | 0.9950 | 0.9482 | 0.9917 | 1.0000 | 0.9958 | Tốt |
| 6 | lemon | 0.9950 | 0.9471 | 0.9939 | 1.0000 | 0.9969 | Tốt |
| 7 | peach | 0.9950 | 0.9395 | 0.9966 | 1.0000 | 0.9983 | Tốt |
| 8 | plum | 0.9950 | 0.9290 | 0.9975 | 1.0000 | 0.9987 | Tốt |
| 9 | sachima | 0.9950 | 0.9259 | 0.9974 | 1.0000 | 0.9987 | Tốt |
| 10 | egg | 0.9950 | 0.9254 | 0.9952 | 1.0000 | 0.9976 | Tốt |
| 11 | litchi | 0.9950 | 0.9183 | 0.9932 | 1.0000 | 0.9966 | Khá |
| 12 | fired_dough_twist | 0.9950 | 0.9172 | 0.9964 | 1.0000 | 0.9982 | Khá |
| 13 | pear | 0.9950 | 0.9120 | 0.9977 | 1.0000 | 0.9988 | Khá |
| 14 | qiwi | 0.9950 | 0.9055 | 0.9973 | 1.0000 | 0.9987 | Khá |
| 15 | bread | 0.9950 | 0.8970 | 0.9949 | 1.0000 | 0.9974 | Yếu — cần cải thiện |
| 16 | mooncake | 0.9950 | 0.8957 | 0.9971 | 1.0000 | 0.9985 | Yếu — cần cải thiện |
| 17 | bun | 0.9950 | 0.8915 | 0.9931 | 1.0000 | 0.9965 | Yếu — cần cải thiện |
| 18 | tomato | 0.9427 | 0.8779 | 0.9520 | 0.9600 | 0.9560 | Yếu — cần cải thiện |
| 19 | coin | 0.9939 | 0.8686 | 0.9983 | 0.9984 | 0.9984 | Yếu — cần cải thiện |
| 20 | banana | 0.9950 | 0.8129 | 1.0000 | 0.9846 | 0.9923 | Yếu — cần cải thiện |

## Phân tích

- **Class xuất sắc** (mAP50-95 ≥ 0.95): 3 class — apple, doughnut, mango
- **Class yếu** (mAP50-95 < 0.90): 6 class — banana, bread, bun, mooncake, tomato, coin

### Hạn chế với từng class yếu

- **banana** (0.8129): hình dạng cong, dễ bị overlap; nhiều góc chụp khác nhau → bbox dự đoán dao động nhiều trên các IoU cao.
- **coin** (0.8686): class quan trọng cho calorie pipeline; recall = 0.998 nhưng precision ở IoU cao thấp → bbox hơi rộng hơn ground truth. Cần review confusion matrix.
- **tomato** (0.8779): số lượng mẫu hơi thấp (4 objects/image), đồng thời dễ nhầm với các loại quả tròn khác (apple, peach).
- **bun** (0.8915), **mooncake** (0.8957): class thiểu số trong dataset (90, 134 images), ít samples → variance lớn hơn.
- **bread** (0.8970), **qiwi** (0.9055): recall = 1.0 nhưng mAP50-95 thấp → bbox dự đoán đúng vị trí nhưng box không khít.

### So sánh với note trước

Trong `ket_qua_danh_gia_note.md` section 4, các class được ghi nhận yếu trên test split là `bread`, `mango`, `pear`, `qiwi`, `coin`. Trên val split:
- `bread` 0.897, `qiwi` 0.905, `coin` 0.869 → trùng khớp với note (đều dưới 0.91).
- `mango` 0.957 và `pear` 0.912 ở val cao hơn đáng kể so với test, cho thấy test split có thể khó hơn (nhiều ảnh đa dạng hơn).