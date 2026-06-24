# 🍽️ Phân tích Pipeline Ước lượng Calo Thực phẩm từ Ảnh RGB đơn: Xác định Bottleneck

> **Báo cáo kết quả Thực tập cơ sở** — Pipeline Analysis of Single-Image Food Calorie Estimation: Identifying the True Bottleneck beyond Detection.

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-%E2%89%A5%202.2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA Version](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![YOLO Version](https://img.shields.io/badge/YOLO-v13n%20%7C%20v8n-FF6F00?logo=ultralytics&logoColor=white)](https://github.com/iMoonLab/YOLOv13)
[![Dataset](https://img.shields.io/badge/Dataset-ECUSTFD-success)](https://github.com/Liang-yc/ECUSTFD-resized-)
[![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux-0078D4?logo=windows&logoColor=white)](https://www.microsoft.com/windows)

---

## 🚀 1. Giới thiệu dự án

Dự án nghiên cứu và xây dựng hệ thống **phát hiện thực phẩm kết hợp ước lượng calories** từ ảnh chụp món ăn, sử dụng các kỹ thuật deep-learning hiện đại:

* 🔍 **Detector:** YOLOv13n (nano) — biến thể nhẹ nhất của dòng YOLOv13, được train lại (fine-tune) trên tập dữ liệu ECUSTFD.
* ⚖️ **Baseline so sánh:** YOLOv8n — dùng để đối chiếu về độ chính xác và tốc độ với YOLOv13n.
* 🧪 **Knowledge Distillation:** chưng cất tri thức từ YOLOv13n (teacher) sang YOLOv8n (student) để có detector nhỏ mà vẫn giữ đúng đắn.
* 📐 **Calorie estimation:** kết hợp bbox detection với hình học đồng xu (coin) làm thước đo tham chiếu, sau đó:
  * 📸 **1-view (top only):** ước lượng thể tích từ silhouette trên cùng + tỉ lệ depth theo class.
  * 📸📸 **2-view (top + side):** ước lượng thể tích L×W×D thật từ hai ảnh top/side → so sánh trực tiếp đóng góp của geometry.
* 📊 **Dataset chính:** [ECUSTFD (Liang & Li, 2017)](https://github.com/Liang-yc/ECUSTFD-resized-) — 19 loại thực phẩm (táo, chuối, bánh mì, bánh bao, doughnut, trứng, grape, lemon, litchi, mango, mooncake, orange, peach, pear, plum, qiwi, sachima, tomato, fired_dough_twist) + 1 class `coin` (đồng xu 25 mm) để calibrate mm/px.
* 📁 **Dữ liệu phụ:** workbook ECUSTFD gốc (`density.xls`) chứa thể tích (cm³) và khối lượng (g) thực tế từng ảnh — dùng làm ground-truth cho bài toán regression calorie.

### 📈 Kết quả chính (Đã chạy sẵn, có thể reproduce)

| Pipeline | n pairs | MAPE (kcal) | Acc@20% |
|---|---:|---:|---:|
| 1-view (top only, GT bbox) | 158 | 48.43 % | 34.18 % |
| **2-view (top + side, GT bbox)** | 158 | **46.53 %** 🏆 | **42.41 %** 🏆 |
| ECUSTFD paper baseline (Liang & Li 2017) | 297 | 18.9 % | — |

<br/>

| Detector | Params | mAP50 | mAP50-95 | Latency (ms) | FPS |
|---|---:|---:|---:|---:|---:|
| yolov8n_baseline | 3.01 M | 0.9908 | 0.9093 | 19.47 | 51.4 |
| **yolov13n_local** | **2.45 M** ⚡ | **0.9923** 🏆 | **0.9173** 🏆 | 34.34 | 29.1 |
| yolov8n_local | 3.01 M | 0.9924 | 0.9162 | 18.27 | 54.7 |
| yolov13 → yolov8n_distilled | 3.01 M | 0.9913 | 0.9124 | 18.93 | 52.8 |

### 📂 Cấu trúc repository

```text
YOLOv13-based Food Detection and Calorie Estimation/
├── 📝 README.md                       ← file này (hướng dẫn tổng)
├── 📁 Documents/
│   ├── 📄 FinalReport/                ← Báo cáo cuối kỳ (PDF)
│   ├── 📄 MidtermReport/              ← Báo cáo giữa kỳ
│   └── 📄 WeeklyReports/              ← Báo cáo tuần
└── 📁 SourceCode/
    ├── 📝 README.md                   ← README chi tiết của source code
    ├── 📋 requirements.txt            ← danh sách package Python cần cài
    ├── 📁 data/
    │   ├── 📊 density.xls             ← workbook ECUSTFD gốc (volume/weight)
    │   └── 📄 density_processed.json  ← JSON đã xử lý từ density.xls
    ├── 📁 datasets/
    │   └── 📁 ECUSTFD/                ← dataset ảnh + labels (XEM MỤC 3.1)
    │       ├── 📄 ecustfd.yaml
    │       ├── 📄 ecustfd_distill.yaml
    │       ├── 📁 images/{train,val,test}/     ← 2978 ảnh JPG
    │       └── 📁 labels/{train,val,test}/     ← YOLO format labels
    ├── 📁 weights/                    ← pretrained + best weights (XEM MỤC 3.2)
    │   ├── 💾 yolov13n_pretrained.pt
    │   ├── 💾 yolov13n_ecustfd_best.pt
    │   ├── 💾 yolov8n_pretrained.pt
    │   └── ...
    ├── 📁 yolov13/                    ← repo YOLOv13 vendored (auto-cài)
    ├── 📁 scripts/                    ← 40+ script train / eval / analyze
    │   ├── 🐍 train_local.py
    │   ├── 🐍 train_yolov8n.py
    │   ├── 🐍 distill_v13_to_v8.py
    │   ├── 🐍 eval_calorie_dual_view_gt.py
    │   ├── 🐍 eval_calorie_dual_view.py
    │   ├── 🐍 calorie_estimator.py
    │   ├── 🐍 check_local_train_ready.py
    │   └── ...
    ├── 📁 runs/                       ← kết quả đã chạy sẵn (tham khảo)
    ├── 💾 best.pt / last.pt           ← best/last weights (sinh ra sau train)
    └── 💾 v13n.onnx / v13n_dynamic.onnx  ← ONNX export (sinh ra sau export)
```

---

## 💻 2. Yêu cầu môi trường

| Thành phần | Yêu cầu | Ghi chú |
|---|---|---|
| **Python** | **3.11** (khuyến nghị) hoặc 3.10 / 3.12 | `check_local_train_ready.py` chỉ chấp nhận 3.10–3.12. Python 3.13 / 3.14 sẽ thiếu wheel cho `pycocotools==2.0.7` và `numpy==1.26.4` (đã test). |
| **GPU (train)** | NVIDIA ≥ 6 GB VRAM | Đã test trên RTX 4050 6 GB. Không bắt buộc cho đánh giá dual-view GT. |
| **CUDA** | 11.8 hoặc 12.1 (khớp với torch wheel) | Chọn wheel tại https://pytorch.org/get-started/locally/ |
| **CPU (eval GT only)**| Bất kỳ | `eval_calorie_dual_view_gt.py` chỉ cần CPU. |
| **RAM** | 8 GB trở lên | 16 GB nếu muốn train. |
| **Disk** | ~5 GB trống (khuyến nghị an toàn) | dataset (~125 MB) + weights (~50 MB) + ONNX/best/last ở root (~30 MB) + pip cache (~1–2 GB) + runs (~60 MB) + dataset khi giải nén. |
| **Git** | ≥ 2.30 | Tùy chọn, dùng khi muốn `git clone` lại. |
| **Hệ điều hành** | Windows 10/11, Linux, WSL2, macOS | Hướng dẫn dưới đây ưu tiên **PowerShell (Windows)**. |

> [!IMPORTANT]
> **Lưu ý về Python:** Nếu máy tính của bạn cài Python 3.14 mặc định (ví dụ `py -0` chỉ liệt kê `-V:3.14`), bạn **bắt buộc phải cài thêm Python 3.11** (xem hướng dẫn ở [Mục 4.1](#41-cai-python-311-windows)) trước khi tiếp tục. Lý do là các thư viện phụ thuộc trong `requirements.txt` chỉ có file wheel được build sẵn hỗ trợ tối đa đến Python 3.12.

---

## 📥 3. Các file KHÔNG có trên GitHub và cách tải về

Vì dung lượng lớn, repository trên GitHub **đã được loại bỏ** các file/thư mục sau. Khi clone về bạn phải tự tải bằng các script hoặc hướng dẫn đính kèm.

| Thư mục / File | Dung lượng | Bắt buộc? | Mục đích |
|---|---:|:---:|---|
| `SourceCode/datasets/ECUSTFD/images/` | ~125 MB | **✔️ Có** | 2978 ảnh JPG (top + side) của ECUSTFD |
| `SourceCode/datasets/ECUSTFD/labels/` | ~285 KB | **✔️ Có** | YOLO-format labels (đã annotate) |
| `SourceCode/weights/*.pt` (8 file) | ~50 MB | **✔️ Có** | Pretrained + best weights để eval/inference ngay |
| `SourceCode/yolov13/` | ~2 MB | **❌ Không** | Bộ skeleton YOLOv13; tự cài qua `pip install ultralytics` |
| `SourceCode/runs/` | ~60 MB | **❌ Không** | Kết quả đã chạy sẵn — chỉ để tham khảo |
| `SourceCode/best.pt`, `last.pt` | ~12 MB | **❌ Không** | Output sau train — script `train_local.py` tự sinh |
| `SourceCode/v13n.onnx`, `v13n_dynamic.onnx` | ~22 MB | **❌ Không** | ONNX export — script `export_tensorrt.py` tự sinh |
| `SourceCode/data/density.xls` | ~46 KB | **❌ Không** | Đã có sẵn `density_processed.json` (đã xử lý) |

### 📂 3.1. Tải dataset ECUSTFD (bắt buộc)

Có **2 cách** để tải dataset — khuyến nghị sử dụng **Cách A**.

#### 💡 Cách A — Dùng `gdown` (Khuyến nghị, nhanh chóng, không cần Git)

```powershell
cd "E:\AI_Research\YOLOv13-based Food Detection and Calorie Estimation\SourceCode"

# Cài gdown (một lần, có thể cài global hoặc trong venv)
python -m pip install gdown

# Tải dataset ECUSTFD đã được chuẩn hoá (ảnh + labels + yaml)
# Link Google Drive folder/file được tác giả đồ án cung cấp — thay FILE_ID bên dưới bằng ID thật.
gdown --folder "https://drive.google.com/drive/folders/ECUSTFD_DATASET_FOLDER_ID" -O datasets/

# Hoặc nếu là 1 file zip duy nhất:
gdown "https://drive.google.com/uc?id=ECUSTFD_DATASET_FILE_ID" -O ecustfd.zip
Expand-Archive -Path ecustfd.zip -DestinationPath datasets\ECUSTFD -Force
```

> [!NOTE]
> **Thay đổi ID thực tế:** Hãy thay thế `ECUSTFD_DATASET_FOLDER_ID` hoặc `ECUSTFD_DATASET_FILE_ID` bằng ID thật do tác giả cung cấp trong link Google Drive. Nếu dùng file chia sẻ công khai ở trạng thái "Bất kỳ ai có liên kết", chỉ cần copy phần ký tự nằm sau `/d/` và trước `/view` của URL.

Sau khi tải xong, hãy kiểm tra tính toàn vẹn của thư mục:

```powershell
Get-ChildItem datasets\ECUSTFD\images
# Kỳ vọng: train/  val/  test/

Get-ChildItem datasets\ECUSTFD\labels
# Kỳ vọng: train/  val/  test/

Get-ChildItem datasets\ECUSTFD\ecustfd.yaml
```

Đếm số lượng ảnh:

```powershell
(Get-ChildItem -Recurse datasets\ECUSTFD\images\train).Count   # Kỳ vọng: 622
(Get-ChildItem -Recurse datasets\ECUSTFD\images\val).Count     # Kỳ vọng: 623
(Get-ChildItem -Recurse datasets\ECUSTFD\images\test).Count    # Kỳ vọng: 1733
```

#### 🛠️ Cách B — Tự tải từ repo gốc ECUSTFD rồi chuẩn hoá

```powershell
# Clone repo dataset gốc (Liang & Li 2017)
git clone https://github.com/Liang-yc/ECUSTFD-resized-.git tmp_ecustfd

# Ảnh gốc đã có sẵn; copy về đúng cấu trúc
Copy-Item -Recurse tmp_ecustfd\data\* datasets\ECUSTFD\images\

# Labels YOLO được tác giả đồ án cung cấp kèm theo — xin qua email/Google Drive.
```

> [!TIP]
> Cách B chỉ phù hợp nếu bạn muốn tái thiết lập (reproduce) quy trình gán nhãn (annotation pipeline). Nếu không, hãy **sử dụng cách A** để tối ưu hóa thời gian.

### 💾 3.2. Tải pretrained + best weights (Bắt buộc nếu không train lại)

Các weights đã được huấn luyện sẵn trên card RTX 4050. Để đánh giá hoặc sử dụng ngay mà không cần train lại, hãy tải về qua lệnh:

```powershell
cd "E:\AI_Research\YOLOv13-based Food Detection and Calorie Estimation\SourceCode"

python -m pip install gdown    # Nếu chưa cài

# Tải cả folder weights (8 file .pt) — thay FOLDER_ID bằng ID thật trên Google Drive
gdown --folder "https://drive.google.com/drive/folders/WEIGHTS_FOLDER_ID" -O weights\
```

Kiểm tra sau khi tải thành công:

```powershell
Get-ChildItem weights\*.pt
# Kỳ vọng có 8 file:
#  yolov13n_pretrained.pt
#  yolov13n_ecustfd_best.pt
#  yolov13n_ecustfd_last.pt
#  yolov8n_pretrained.pt
#  yolov8n_ecustfd_best.pt
#  yolov8n_ecustfd_last.pt
#  yolov8n_distilled_best.pt
#  yolov8n_distilled_last.pt
```

### ⚙️ 3.3. Tải repo YOLOv13 (Không bắt buộc — script tự cài)

Thư mục `yolov13/` chỉ chứa khung xương (skeleton) + file cấu hình `pyproject.toml`. Khi bạn chạy lệnh cài đặt ở [Mục 4.4](#44-cai-cac-package-con-lai--yolov13), hệ thống `pip` sẽ tự động pull các thư viện phụ thuộc (`ultralytics ≥ 8.3`) từ PyPI. Nếu gặp lỗi do thiếu thư mục `yolov13/` trong repo, hãy chạy lệnh dưới đây để clone thủ công:

```powershell
git clone https://github.com/iMoonLab/YOLOv13.git yolov13
```

---

## 🛠️ 4. Hướng dẫn cài đặt

### 🐍 4.1. Cài đặt Python 3.11 (Windows)

**Cách 1: Tải bộ cài installer từ python.org (Khuyến nghị, ổn định nhất):**

1. Truy cập trang phát hành chính thức: https://www.python.org/downloads/release/python-3119/ (Bản **3.11.9** rất ổn định).
2. Tải file cài đặt **Windows installer (64-bit)**.
3. Chạy file cài đặt, lưu ý **tick chọn vào mục “Add Python 3.11 to PATH”** trước khi bấm **Install Now**.
4. Kiểm tra lại phiên bản sau khi cài đặt:

   ```powershell
   py -0
   # Kỳ vọng danh sách hiển thị:
   #  -V:3.11 *        Python 3.11 (64-bit)
   #  -V:3.14 *        Python 3.14 (64-bit)
   ```

**Cách 2: Cài qua winget (PowerShell, nếu máy có sẵn Windows Package Manager):**

```powershell
winget install --id Python.Python.3.11 -e
py -0
```

### 📦 4.2. Tạo môi trường ảo (Virtual Environment)

```powershell
cd "E:\AI_Research\YOLOv13-based Food Detection and Calorie Estimation\SourceCode"

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Nếu hệ thống báo lỗi chính sách bảo mật PowerShell, hãy chạy lệnh sau:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
```

> [!NOTE]
> Kiểm tra dòng lệnh PowerShell của bạn phải hiển thị tiền tố `(.venv)` dạng: `(.venv) PS E:\...\SourceCode>`.

### 🔥 4.3. Cài đặt PyTorch (Khớp đúng phiên bản CUDA)

Truy cập https://pytorch.org/get-started/locally/ để lấy lệnh cài đặt phù hợp nhất với card NVIDIA hiện tại. Dưới đây là hai lệnh phổ biến đã được kiểm nghiệm:

```powershell
# --- CUDA 12.1 (Khuyến nghị cho driver card đồ họa đời mới) ---
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# --- CUDA 11.8 (Dành cho driver card đồ họa đời cũ hơn) ---
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

Kiểm tra xem PyTorch đã nhận dạng CUDA và card đồ họa hay chưa:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### ⚡ 4.4. Cài các package còn lại + YOLOv13

```powershell
pip install -r requirements.txt
pip install -e yolov13 --no-deps --no-build-isolation
```

> [!TIP]
> Lệnh thứ hai dùng để cài đặt repository YOLOv13 ở chế độ có thể chỉnh sửa (`editable`) mà không cần build wheel thủ công.

Nếu gặp lỗi trong quá trình cài đặt:

```powershell
# Lỗi "ERROR: yolov13/ ... file not found" (do chưa clone thư mục yolov13/)
git clone https://github.com/iMoonLab/YOLOv13.git yolov13
pip install -e yolov13 --no-deps --no-build-isolation

# Lỗi hiếm gặp liên quan đến Cython/Visual Studio Build Tools
pip install cython numpy
```

### 🔌 4.5. Cài các công cụ hỗ trợ phụ

```powershell
pip install gdown         # Sử dụng để tải dữ liệu (Mục 3.1 & 3.2)
# Chỉ cài nếu cần chạy xuất mô hình ONNX/TensorRT. Lưu ý: bản onnxruntime-gpu phải trùng khớp với bản CUDA của torch.
pip install onnx onnxruntime onnxruntime-gpu onnxslim
```

> [!WARNING]
> **Cảnh báo xung đột onnxruntime-gpu:** Nếu bạn cài đặt `onnxruntime-gpu` rồi sau đó chạy lệnh cài `onnxruntime` (bản CPU), pip sẽ tự động gỡ cài đặt phiên bản GPU. Vui lòng chỉ chọn cài 1 trong 2 thư viện này.

---

## 🔍 5. Kiểm tra môi trường

Hãy chạy script kiểm tra tự động để đảm bảo môi trường đã sẵn sàng cho việc training:

```powershell
cd "E:\AI_Research\YOLOv13-based Food Detection and Calorie Estimation\SourceCode"
python scripts/check_local_train_ready.py
```

Kỳ vọng kết quả đầu ra:

```text
Status: LOCAL_TRAIN_READY
Report: datasets\ECUSTFD\reports\local_train_readiness_report.json
```

> [!IMPORTANT]
> - Nếu kết quả không xuất hiện bất kỳ dòng nào bắt đầu bằng ký tự `- ` thì môi trường hoàn toàn ổn định (không có blocker).
> - Nếu máy tính **không có GPU/CUDA**, script sẽ hiển thị cảnh báo `- CUDA is not available from torch.` và trạng thái `LOCAL_TRAIN_NOT_READY`. Đây là thiết kế đúng (train cần GPU, còn nếu chỉ chạy đánh giá dual-view ground-truth ở [Mục 6.1](#61-chay-nhanh--danh-gia-calorie-dual-view-bang-ground-truth-bbox-khong-can-gpu-5-giay-tong) thì **không yêu cầu GPU/CUDA**).

Báo cáo chi tiết dạng JSON lưu trữ thông tin về:
- Phiên bản Python (`supported_for_torch: true` đối với 3.10–3.12).
- Trạng thái CUDA và tên GPU nhận diện được.
- Trạng thái kiểm tra các module bắt buộc (`torch`, `ultralytics`, `cv2`, `PIL`, `thop`, `timm`, `safetensors`, `huggingface_hub`).
- Trạng thái kiểm tra file hệ thống (`datasets/ECUSTFD/ecustfd.yaml`, `weights/yolov13n_pretrained.pt`, `yolov13/ultralytics/__init__.py`).

---

## 📊 6. Chạy chương trình

> [!NOTE]
> Tất cả các lệnh dưới đây giả định bạn đang đứng trong thư mục `SourceCode\` và môi trường ảo `.venv` đã được kích hoạt thành công (`Activate.ps1`).

### ⚡ 6.1. Chạy nhanh — Đánh giá calorie dual-view bằng ground-truth bbox (Không cần GPU, chạy < 5 giây)

Đây chính là luồng xử lý (pipeline) **đã được sử dụng trong báo cáo tốt nghiệp**:

```powershell
python scripts/eval_calorie_dual_view_gt.py ^
    --labels-root datasets\ECUSTFD\labels ^
    --density-json data\density_processed.json ^
    --output runs\dual_view_eval_gt
```

Hoặc sử dụng file chạy tự động (tương đương):

```powershell
scripts\run_dual_view_eval_gt.bat
```

Các file kết quả đầu ra sẽ nằm tại thư mục `runs\dual_view_eval_gt\`:
- `summary.json` — Tổng hợp toàn bộ metric dưới dạng file JSON.
- `comparison_1v_vs_2v.md` — Bảng so sánh trực quan phục vụ báo cáo.
- `comparison_1v_vs_2v.csv` — File CSV chứa dữ liệu so sánh.
- `per_class_metrics_v1.csv` & `per_class_metrics_v2.csv` — Chi tiết metric cho từng class.
- `per_image_predictions.csv` — Dự đoán chi tiết trên từng ảnh chụp.
- `error_analysis.md` — Phân tích chi tiết các trường hợp sai lệch.

### 🖼️ 6.2. Chạy thử nghiệm detector (Sample Inference) trên một số ảnh mẫu

```powershell
python scripts\run_sample_inference.py ^
    --source datasets\ECUSTFD\images\test ^
    --weights weights\yolov13n_ecustfd_best.pt ^
    --output runs\local_food_detect\sample_inference ^
    --max-images 8
```

Kết quả xuất ra tại thư mục chỉ định:
- `runs\local_food_detect\sample_inference\*_pred.jpg` — Ảnh kết quả đã vẽ sẵn bounding box.
- `runs\local_food_detect\sample_inference\detections.csv` — Thống kê chi tiết các đối tượng phát hiện được (mỗi dòng là một detection).
- `runs\local_food_detect\sample_inference\detections.json` — File JSON lưu dữ liệu phát hiện.
- `runs\local_food_detect\sample_inference\summary.json` — Tổng hợp số lượng detection, phân bố class, sự hiện diện của đồng xu (`coin`) trên từng ảnh.

### 🏋️ 6.3. Huấn luyện YOLOv13n từ đầu (Yêu cầu GPU ≥ 6 GB VRAM, mất vài giờ)

```powershell
python scripts\train_local.py --batch 4 --workers 2 --imgsz 640 --epochs 100
```

**Các kết quả tự động sinh ra bởi script `train_local.py`:**
- `runs\local_food_detect\yolov13n_ecustfd_local\weights\best.pt` — Trọng số tốt nhất.
- `runs\local_food_detect\yolov13n_ecustfd_local\weights\last.pt` — Trọng số ở epoch cuối cùng.
- `runs\local_food_detect\yolov13n_ecustfd_local\results.csv` — Lịch sử chỉ số qua các epochs.
- `runs\local_food_detect\yolov13n_ecustfd_local\results.png` & `confusion_matrix.png` — Đồ thị kết quả và ma trận nhầm lẫn.
- `confusion_matrix_normalized.png`, `labels.jpg`, `labels_correlogram.jpg` — Biểu đồ phân tích nhãn.
- **Thư mục đồng bộ tự động:** Sao chép song song kết quả sang `runs\local_food_detect\output\` (gồm weights, result.csv và confusion matrix).

> [!NOTE]
> File tổng hợp chỉ số `runs\local_food_detect\output\metrics_summary.md` trong repo gốc được tạo thủ công từ log training để phục vụ báo cáo tốt nghiệp, không phải do script tự sinh.

### 🧪 6.4. Huấn luyện baseline YOLOv8n

```powershell
python scripts\train_yolov8n.py --batch 4 --workers 2 --imgsz 640 --epochs 100
```

### 🤝 6.5. Chưng cất tri thức (Knowledge Distillation) từ YOLOv13n sang YOLOv8n

```powershell
python scripts\distill_v13_to_v8.py      # Hiển thị kế hoạch và thiết lập teacher/student (dạng mã giả)
python scripts\generate_pseudo_labels.py # Tạo nhãn giả (pseudo-labels) từ YOLOv13n trên tập train
python scripts\train_baseline_v8n.py     # Huấn luyện mô hình YOLOv8n baseline với nhãn chuẩn GT
python scripts\train_student.py          # Huấn luyện mô hình học trò YOLOv8n với nhãn gộp (GT + pseudo)
```

> [!NOTE]
> Script `distill_v13_to_v8.py` chỉ in ra mã giả kế hoạch chạy. Quá trình distillation thực tế được thực hiện offline qua việc gán nhãn giả (xem chi tiết ở script `generate_pseudo_labels.py` + `merge_labels.py` trước khi chạy train student).

### 📐 6.6. Đánh giá ước lượng calorie dual-view với detector YOLOv13n thật (Cần GPU + best.pt)

```powershell
python scripts\eval_calorie_dual_view.py ^
    --source datasets\ECUSTFD\images\test ^
    --weights weights\yolov13n_ecustfd_best.pt ^
    --density-json data\density_processed.json ^
    --output runs\dual_view_eval
```

### 🔄 6.7. Tái tạo dữ liệu `density_processed.json` từ file Excel gốc `density.xls` (Tùy chọn)

```powershell
python scripts\parse_density_xls.py --xls data\density.xls --out data\density_processed.json
```

---

## 📝 7. Kết quả & tham khảo

Toàn bộ các thư mục kết quả đã được chạy và lưu sẵn phục vụ bảng kết quả trong báo cáo (paper Table) nằm tại thư mục `runs/`:

| Thư mục kết quả | Ý nghĩa / Nội dung lưu trữ |
|---|---|
| `runs/dual_view_eval_gt/` | **🏆 Bảng số liệu chính** — So sánh phương pháp 1-view và 2-view trên 158 cặp ảnh |
| `runs/calorie_eval/` | Đánh giá toàn bộ pipeline YOLOv13n trên 1733 ảnh test (1728 ảnh nhận diện thành công, 5 ảnh không có thực phẩm) |
| `runs/calorie_eval_yolov8n/` | Đánh giá toàn bộ pipeline YOLOv8n baseline (1724 ảnh nhận diện thành công trên 1733 ảnh test, 9 ảnh không có thực phẩm) |
| `runs/calorie_ablation_v13/` | Thử nghiệm loại bỏ (ablation) các chính sách tính density (per-image / per-class / fallback) |
| `runs/distill_v13_to_v8/` | Dữ liệu liên quan đến chưng cất tri thức (Knowledge Distillation) YOLOv13n → YOLOv8n |
| `runs/midas_calorie/` | Kết quả dự đoán bản đồ độ sâu từ mô hình MiDaS (`per_image_midas.csv`) so sánh với tỷ lệ depth cố định |
| `runs/v13_speedup/` | Kết quả benchmark đo tốc độ YOLOv13n định dạng ONNX và TensorRT |
| `runs/v13_vs_v8_detailed/` | So sánh chi tiết hiệu năng giữa YOLOv13 và YOLOv8 |
| `runs/kd_benchmark/` | So sánh latency (độ trễ) giữa teacher (YOLOv13n), student baseline và distilled student |
| `runs/statistical_analysis/` | Các phép kiểm định thống kê ý nghĩa sự khác biệt (significance test) và breakdown theo từng class |

> [!TIP]
> Bạn có thể xem chi tiết tài liệu hướng dẫn chuyên sâu hơn tại file [SourceCode/README.md](file:///E:/AI_Research/YOLOv13-based%20Food%20Detection%20and%20Calorie%20Estimation/SourceCode/README.md) hoặc xem báo cáo PDF hoàn chỉnh tại [Documents/FinalReport/](file:///E:/AI_Research/YOLOv13-based%20Food%20Detection%20and%20Calorie%20Estimation/Documents/FinalReport/).

---

## ⚠️ 8. Ghi chú & khắc phục sự cố (Troubleshooting)

* **Không dùng Python 3.13 hoặc 3.14:** Thư viện `pycocotools==2.0.7` và `numpy==1.26.4` không có file cài đặt sẵn (wheel prebuilt) cho các bản Python này; cài đặt từ source code sẽ cực kỳ dễ phát sinh lỗi build. Hãy dùng **Python 3.11**.
* **Lỗi không nhận lệnh `py -3.11`:** Trình khởi chạy Python trên Windows chỉ nhận diện các phiên bản Python cài bằng bộ cài (installer) chính thức tải từ python.org. Vui lòng cài đặt lại như hướng dẫn ở [Mục 4.1](#41-cai-python-311-windows).
* **PowerShell chặn file kích hoạt môi trường ảo (`Activate.ps1`):** Hãy chạy lệnh `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force` bằng quyền Admin/User hiện tại rồi thử kích hoạt lại.
* **Mô hình không nhận card đồ họa (CUDA):** Kiểm tra xem driver card đồ họa NVIDIA đã được cài đặt chưa bằng lệnh `nvidia-smi`. Phiên bản CUDA của PyTorch cần tương thích với driver NVIDIA của bạn (không yêu cầu máy phải cài đặt sẵn CUDA Toolkit độc lập).
* **Thiếu thư mục `yolov13/`:** Chạy lệnh `git clone https://github.com/iMoonLab/YOLOv13.git yolov13` rồi thực hiện lại lệnh `pip install -e yolov13 --no-deps --no-build-isolation`.
* **Không tìm thấy ảnh hoặc nhãn dữ liệu:** Hãy chạy lại các bước tải dữ liệu ở [Mục 3.1](#31-tai-dataset-ecustfd-bat-buoc).
* **Không tìm thấy trọng số (weights):** Hãy tải lại các weights theo hướng dẫn ở [Mục 3.2](#32-tai-pretrained--best-weights-bat-buoc-neu-khong-train-lai).
* **Báo cáo log bỏ qua 2 ảnh training:** Hai ảnh `mix002T(2)` và `mix005S(4)` bị lỗi calibration (đồng xu tham chiếu) từ tập dữ liệu ECUSTFD gốc nên script tự động bỏ qua nhờ biến `INVALID_CALIBRATION_STEMS`. Đây là hành vi hoạt động hoàn toàn chính xác.
* **Lỗi đường dẫn file:** Mọi script trong dự án đều thiết kế sử dụng đường dẫn tương đối. Bạn bắt buộc phải di chuyển con trỏ dòng lệnh vào thư mục `SourceCode\` trước khi chạy bất kỳ script nào.
* **Lỗi tràn bộ nhớ đồ họa (OOM) khi train:** Hãy giảm kích thước batch size `--batch 4` xuống `--batch 2` hoặc hạ kích thước ảnh `--imgsz 640` xuống `--imgsz 512`.

---

## 📚 9. Tài liệu tham khảo

* Mô hình & paper YOLOv13: [iMoonLab/YOLOv13](https://github.com/iMoonLab/YOLOv13)
* Tập dữ liệu ECUSTFD gốc: [Liang-yc/ECUSTFD-resized-](https://github.com/Liang-yc/ECUSTFD-resized-) *(Nguồn: Liang & Li, "ECUSTFD — A Large-Scale Database for Food Image Recognition", 2017)*
* Thư viện Ultralytics: [Ultralytics Docs](https://docs.ultralytics.com/)
* Thư viện PyTorch: [PyTorch Homepage](https://pytorch.org/)

---

* 🧑‍💻 **Tác giả:** Đặng Văn Chiến - B23DCCE012 — Báo cáo kết quả Thực tập cơ sở, ngành Công nghệ thông tin.
* 👨‍🏫 **GVHD:** Thầy Kim Ngọc Bách.
* 📅 **Năm thực hiện:** 2025–2026.
