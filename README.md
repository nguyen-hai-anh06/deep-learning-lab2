# Practice 2: Pre-trained Neural Network Architectures on CIFAR-10

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyen-hai-anh06/deep-learning-lab2/blob/main/practice2_cv_colab.ipynb)

Báo cáo và mã nguồn thực nghiệm so sánh 4 kiến trúc mạng nơ-ron học sâu (Deep Convolutional Neural Networks) tiền huấn luyện (Pre-trained on ImageNet-1k) áp dụng kỹ thuật **Transfer Learning (Feature Extraction / Fine-Tuning)** trên tập dữ liệu **CIFAR-10**.

## 📌 Các kiến trúc khảo sát

1. **ResNet-18** (Residual Networks - Deep Residual Learning)
2. **VGG-16** (Visual Geometry Group)
3. **DenseNet-121** (Densely Connected Convolutional Networks)
4. **MobileNetV2** (Inverted Residuals and Linear Bottlenecks)

---

## 📊 Kết quả thực nghiệm (Evaluation Results)

Kết quả đánh giá trên tập kiểm thử (10,000 ảnh CIFAR-10 test set) với chế độ đóng băng backbone (Freeze Backbone, chỉ huấn luyện classifier):

| Mô hình              | Test Accuracy (%) | Test Loss        | Tham số (Total) | Tham số huấn luyện (Trainable) | Dung lượng mô hình (MB) | Độ trễ (ms/batch) |
| ---------------------- | ----------------- | ---------------- | ---------------- | --------------------------------- | --------------------------- | -------------------- |
| **DenseNet-121** | **82.28%**  | **0.5221** | 6.96 M           | 10.25 K                           | 26.57 MB                    | 181.44 ms            |
| **VGG-16**       | 81.30%            | 0.5505           | 134.30 M         | 40.97 K                           | 512.32 MB                   | 289.85 ms            |
| **ResNet-18**    | 81.26%            | 0.5550           | 11.18 M          | 5.13 K                            | 42.65 MB                    | **51.96 ms**   |
| **MobileNetV2**  | 75.97%            | 0.7076           | **2.24 M** | 12.81 K                           | **8.53 MB**           | 56.54 ms             |

> Xem phân tích chi tiết và kịch bản thuyết trình tại [presentation.md](presentation.md).

---

## 📁 Cấu trúc thư mục (Project Structure)

```text
├── lab2_cv/                   # Core package chứa cấu hình và các services
│   ├── config.py              # Tham số cấu hình, đường dẫn, hyper-parameters
│   └── services/
│       ├── data_service.py    # Pipeline nạp dữ liệu CIFAR-10, transforms, dataloaders
│       ├── model_service.py   # Khởi tạo mô hình, đóng băng backbone, inspect kiến trúc
│       ├── trainer_service.py # Huấn luyện, đánh giá, logging metrics
│       └── logger_service.py  # Ghi log TensorBoard & console
├── tests/                     # Unit tests kiểm thử hệ thống
├── results/                   # Kết quả đánh giá và tóm tắt huấn luyện dạng JSON
│   ├── evaluation_results.json
│   └── training_summary.json
├── train.py                   # Script CLI huấn luyện mô hình
├── run_evaluation.py          # Script CLI đánh giá benchmark các mô hình
├── practice2_cv_colab.ipynb   # Jupyter Notebook chạy thực nghiệm trên Google Colab
├── presentation.md            # Báo cáo chi tiết và tài liệu thuyết trình
├── requirements.txt           # Danh sách các thư viện phụ thuộc
└── .gitignore                 # Cấu hình bỏ qua dữ liệu nặng và checkpoints
```

---

## 🚀 Cài đặt & Hướng dẫn sử dụng

### 0. Chạy trên Google Colab qua GitHub & Google Drive

- **Cách 1 (Mở nhanh 1-Click)**: Bấm trực tiếp vào huy hiệu [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyen-hai-anh06/deep-learning-lab2/blob/main/practice2_cv_colab.ipynb) để mở notebook trên Colab.
- **Cách 2 (Khuyên dùng - Clone vào Google Drive để lưu weights vĩnh viễn)**:
  1. Mở Google Colab, chọn Runtime GPU T4 (**Runtime** -> **Change runtime type** -> **T4 GPU**).
  2. Tạo 1 ô code để mount Google Drive và clone dự án từ GitHub:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     %cd /content/drive/MyDrive
     !git clone https://github.com/nguyen-hai-anh06/deep-learning-lab2.git
     %cd deep-learning-lab2
     ```
  3. Mở file `practice2_cv_colab.ipynb` từ Google Drive để chạy thực nghiệm.

### 1. Cài đặt môi trường (Local máy cá nhân)

Khuyến nghị sử dụng Python 3.10+:

```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình

- Huấn luyện một mô hình cụ thể (ví dụ: `resnet18`):

```bash
python train.py --model resnet18 --epochs 10 --batch_size 64
```

- Huấn luyện toàn bộ 4 mô hình:

```bash
python train.py --model all --epochs 10
```

- Chạy thử nghiệm nhanh (Dry-run với tập mẫu nhỏ 500 ảnh):

```bash
python train.py --model mobilenet_v2 --subset 500 --epochs 2
```

### 3. Đánh giá mô hình

```bash
python run_evaluation.py
```

### 4. Giám sát với TensorBoard

```bash
tensorboard --logdir=runs
```

---

## ⚙️ Lưu ý về Checkpoints & Data

- Dữ liệu tập `cifar-10` và các file checkpoint trọng số mô hình (`checkpoints/*.pth`) có dung lượng lớn (đặc biệt VGG-16 > 500MB) vượt quá giới hạn lưu trữ của GitHub (>100MB), do đó đã được khai báo loại trừ trong `.gitignore`.
- Bạn có thể tải lại dataset tự động bằng cách chạy `train.py` hoặc notebook `practice2_cv_colab.ipynb` trên Google Colab GPU.

---

## 👤 Tác giả thực hiện

- **Họ và tên**: Nguyễn Hải Anh - UTH
- **GitHub**: [@nguyen-hai-anh06](https://github.com/nguyen-hai-anh06)
- **Dự án**: Practice 2 - Pre-trained Neural Network Architectures
