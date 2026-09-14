# Practice 2 — Báo Cáo Kết Quả Thực Nghiệm

> **Trạng thái:** Đã tổng hợp và điền đầy đủ số liệu thực nghiệm từ 8 run trong `Lab2_Experiments` (gồm cấu hình tham số, kết quả validation của 4 mô hình $\times$ 2 chiến lược).

## 1. Mô hình

- ResNet-18, VGG-16 và DenseNet-121 từ `torchvision.models`.
- MobileNetV4 Conv Small từ `timm`: `mobilenetv4_conv_small.e2400_r224_in1k`.
- Tất cả dùng trọng số pre-trained ImageNet-1K và classifier 10 đầu ra cho CIFAR-10.

## 2. Dữ liệu

- Train: 45.000 ảnh, có resize, augmentation và ImageNet normalization.
- Validation: 5.000 ảnh, dùng để chọn checkpoint và hyperparameter.
- Test: 10.000 ảnh, chỉ dùng sau khi chốt run tốt nhất bằng validation.
- Các thành viên dùng seed 42 và kiểm tra `validation_indices_checksum` trong `split.json`.

## 3. Chiến lược

- Freeze: đóng băng backbone, chỉ train classifier.
- Fine-tune last: mở classifier và block cuối phù hợp với từng kiến trúc.
- Loss: CrossEntropyLoss.
- Optimizer, learning rate, batch size, epoch và weight decay lấy từ `config.json` của từng run.

### 3.1. Cấu trúc tổng thể dự án Lab 2 & Chức năng các thành phần

```text
deep-learning-lab2/
├── configs/                          # Cấu hình phân công thí nghiệm cho từng thành viên
│   ├── trainer_a.json                # Phân công Trainer A (VGG-16, MobileNetV4)
│   └── trainer_b.json                # Phân công Trainer B (ResNet-18, DenseNet-121)
├── lab2_cv/                          # Package mã nguồn cốt lõi của dự án
│   ├── config.py                     # Cấu hình siêu tham số, thiết bị (GPU/CPU), hằng số
│   └── services/                     # Các module dịch vụ xử lý nghiệp vụ
│       ├── data_service.py           # Pipeline nạp dữ liệu CIFAR-10, transforms, split train/val/test
│       ├── model_service.py          # Khởi tạo 4 mô hình, quản lý đóng băng (freeze) và fine-tune
│       ├── trainer_service.py        # Vòng lặp huấn luyện, validation, checkpoint atomic, StepLR
│       └── logger_service.py         # Ghi log đồ thị và chỉ số sang TensorBoard
├── Lab2_Experiments/                 # Lưu trữ toàn bộ dữ liệu 8 run thực nghiệm đã hoàn thành
│   ├── trainer_a/                    # 4 run của Trainer A (Freeze & Fine-tune VGG-16, MobileNetV4)
│   ├── trainer_b/                    # 4 run của Trainer B (Freeze & Fine-tune ResNet-18, DenseNet-121)
│   └── validation_summary/           # Bảng kết quả tổng hợp Validation sau khi train
├── results/                          # Thư mục lưu kết quả xuất ra
│   └── visualizations/               # Chứa 8 biểu đồ học thuật (DPI 300) và báo cáo tóm tắt
├── tests/                            # Bộ kiểm thử tự động (Unit tests)
│   ├── test_data.py                  # Kiểm thử tính toàn vẹn của pipeline dữ liệu và transforms
│   └── test_models.py                # Kiểm thử forward pass và logic đóng băng/mở khóa trọng số
├── train.py                          # Script huấn luyện 1 mô hình độc lập (hỗ trợ tự động resume)
├── run_assignment.py                 # Tự động thực thi toàn bộ hàng đợi thí nghiệm được phân công
├── collect_results.py                # Thu thập & so sánh kết quả validation từ 2 trainer
├── run_evaluation.py                 # Đánh giá 1 lần duy nhất trên tập Test (10.000 ảnh) cho 4 model tốt nhất
├── visualize_experiments.py          # Tự động tạo 8 biểu đồ trực quan hóa và báo cáo Markdown
├── team_training_colab.ipynb         # Notebook chạy toàn bộ quy trình trên Google Colab
├── REPORT.md                         # Báo cáo thực nghiệm khoa học chi tiết bằng tiếng Việt
└── presentation.md                   # Khung dàn ý báo cáo kết quả và thuyết trình
```

![Cấu trúc và chức năng toàn bộ dự án Lab 2](results/visualizations/project_structure.png)

## 4. Kiến trúc và tham số

Bảng tổng hợp từ `architecture.txt` và `config.json` của 4 kiến trúc:

| Model | Classification layer | Total params | Trainable freeze | Trainable fine-tune | Model Size (MB) |
|---|---|---:|---:|---:|---:|
| ResNet-18 | `fc` (Linear 512 -> 10) | 11,181,642 | 5,130 (0.05%) | 8,398,858 (75.11%) | 42.66 MB |
| VGG-16 | `classifier[6]` (Linear 4096 -> 10) | 134,301,514 | 40,970 (0.03%) | 7,120,394 (5.30%) | 512.32 MB |
| DenseNet-121 | `classifier` (Linear 1024 -> 10) | 6,964,106 | 10,250 (0.15%) | 2,170,378 (31.17%) | 26.57 MB |
| MobileNetV4 | `classifier` qua `get_classifier()` (Linear 1280 -> 10) | 2,505,834 | 12,810 (0.51%) | 1,366,410 (54.53%) | 9.56 MB |

> *Ghi chú:* Ở chiến lược **Freeze**, chỉ tầng phân loại FC cuối cùng được cập nhật trọng số. Ở chiến lược **Fine-tune**, tầng phân loại cùng các khối tích chập cuối cùng (Layer 4 của ResNet, Block 5 của VGG, DenseBlock 4 + Norm 5 của DenseNet, Conv Head + Block cuối của MobileNetV4) được mở khóa huấn luyện.

## 5. Kết quả validation thực nghiệm (Đã gom từ 8 run của trainer_a và trainer_b)

### 5.1. Bảng so sánh chi tiết giữa hai chiến lược trên tập Validation (5.000 ảnh):

| Model | Strategy | Val accuracy | Best Epoch | Final Train Loss | Final Val Loss | Train time (GPU T4) | Selected for Test |
|---|---|---:|---:|---:|---:|---:|:---:|
| **ResNet-18** | **finetune_last** | **95.22%** | 9 | 0.0264 | 0.1583 | 18.1 phút (1087.5s) | **YES** |
| ResNet-18 | freeze | 86.46% | 5 | 0.4497 | 0.4078 | 17.5 phút (1052.2s) | |
| **DenseNet-121** | **finetune_last** | **94.40%** | 8 | 0.0617 | 0.1631 | 24.4 phút (1465.4s) | **YES** |
| DenseNet-121 | freeze | 88.62% | 7 | 0.3540 | 0.3669 | 21.7 phút (1299.4s) | |
| **VGG-16** | **finetune_last** | **93.86%** | 9 | 0.0705 | 0.2015 | 29.3 phút (1757.5s) | **YES** |
| VGG-16 | freeze | 82.20% | 7 | 0.8124 | 0.5372 | 26.9 phút (1612.1s) | |
| **MobileNetV4** | **finetune_last** | **92.32%** | 10 | 0.1345 | 0.2604 | 18.3 phút (1096.5s) | **YES** |
| MobileNetV4 | freeze | 87.96% | 9 | 0.4190 | 0.3808 | 18.0 phút (1079.6s) | |

> Toàn bộ 4 mô hình đều đạt hiệu năng vượt trội khi chọn chiến lược **finetune_last** để tiến hành đánh giá tập Test.

### 5.2. Kết quả đánh giá kiểm thử trên tập Test (10.000 ảnh)

Bảng tổng hợp từ `results/final_evaluation/test_results.csv` sau khi chạy `run_evaluation.py` (hoặc Cell 5 trên Google Colab):

| Model | Strategy được chọn | Val accuracy | Test accuracy | Test loss | Macro F1 | Train time |
|---|---|---:|---:|---:|---:|---:|
| **ResNet-18** | finetune_last | 95.22% | *Đang cập nhật từ Colab Cell 5* | *Đang cập nhật* | *Đang cập nhật* | 18.1 phút |
| **DenseNet-121** | finetune_last | 94.40% | *Đang cập nhật từ Colab Cell 5* | *Đang cập nhật* | *Đang cập nhật* | 24.4 phút |
| **VGG-16** | finetune_last | 93.86% | *Đang cập nhật từ Colab Cell 5* | *Đang cập nhật* | *Đang cập nhật* | 29.3 phút |
| **MobileNetV4** | finetune_last | 92.32% | *Đang cập nhật từ Colab Cell 5* | *Đang cập nhật* | *Đang cập nhật* | 18.3 phút |

## 6. Nhận xét & Đánh giá khoa học

### 6.1. So sánh độ chính xác với số tham số và hiệu năng tính toán
- **ResNet-18** đạt độ chính xác cao nhất (**95.22%**) đồng thời có thời gian huấn luyện nhanh nhất (**18.1 phút**). Kiến trúc Residual Connections giúp gradient lan truyền hiệu quả, hội tụ cực nhanh mà không cần quá nhiều tham số (11.18M).
- **MobileNetV4 Conv Small** là mô hình **tối ưu tài nguyên nhất**: với vỏn vẹn **2.51M tham số** (~9.56 MB, nhỏ hơn VGG-16 tới **53.6 lần**), mô hình vẫn đạt độ chính xác ấn tượng **92.32%**, rất phù hợp triển khai trên thiết bị nhúng và di động (Edge/Mobile devices).
- **DenseNet-121** (6.96M tham số) đạt **94.40%**, đứng thứ 2 về độ chính xác nhờ cơ chế tái sử dụng đặc trưng liên tục (Dense connectivity), tuy nhiên thời gian huấn luyện (24.4 phút) lâu hơn ResNet-18 do việc gom ghép tensor (feature concatenation) ngốn nhiều băng thông bộ nhớ.
- **VGG-16** (134.3M tham số) là mô hình cồng kềnh nhất (512 MB) và tốn nhiều thời gian nhất (29.3 phút), nhưng độ chính xác (93.86%) lại thấp hơn ResNet-18 và DenseNet-121. Điều này phản ánh rõ hạn chế của kiến trúc tích chập thuần túy không có skip connections.

### 6.2. So sánh chiến lược Freeze Backbone vs Fine-tune Last Layers
- Chiến lược Fine-tune tầng cuối mang lại sự bứt phá vượt bậc:
  - **VGG-16**: tăng mạnh nhất **+11.66%** (82.20% -> 93.86%).
  - **ResNet-18**: tăng **+8.76%** (86.46% -> 95.22%).
  - **DenseNet-121**: tăng **+5.78%** (88.62% -> 94.40%).
  - **MobileNetV4**: tăng **+4.36%** (87.96% -> 92.32%).
- **Lý do**: CIFAR-10 có kích thước ảnh gốc 32x32 (được resize lên 224x224), phân phối đặc trưng có độ phân giải thấp hơn so với ảnh ImageNet thực tế. Việc đóng băng toàn bộ backbone khiến mô hình bị hạn chế (underfitting) vào các đặc trưng cố định. Khi unfreeze block conv cuối, các bộ lọc thích ứng linh hoạt hơn với ngữ cảnh đối tượng trong CIFAR-10.

### 6.3. Quá trình hội tụ và ảnh hưởng của Learning Rate Scheduler
- Tất cả các run sử dụng `StepLR` với `step_size=5, gamma=0.1`.
- Tại epoch 5, Learning Rate giảm 10 lần (từ 1e-4 xuống 1e-5 đối với Fine-tune). Trên đồ thị loss và accuracy, ngay tại epoch 5-6 xuất hiện một bước giảm đột ngột của Validation Loss và Validation Accuracy tăng vọt thêm 1.5 - 2%, giúp mô hình nhanh chóng ổn định và không bị dao động gradient ở các epoch cuối.
- Dấu hiệu Overfitting: Ở chiến lược Fine-tune, Train Accuracy đạt xấp xỉ 97-99% trong khi Val Accuracy đạt 92-95% (gap khoảng 3-4%), là mức kiểm soát overfitting tốt nhờ L2 Regularization (`weight_decay=1e-4`) và Data Augmentation (RandomHorizontalFlip, RandomRotation).

---

## 7. Thư viện biểu đồ phục vụ báo cáo

Các biểu đồ độ phân giải cao đã được tạo sẵn tại `results/visualizations/`:
1. `results/visualizations/strategy_gain_comparison.png` — So sánh tăng trưởng độ chính xác giữa Freeze và Fine-tune.
2. `results/visualizations/loss_curves_by_model.png` — Đường cong hàm mất mát (Loss curves) 4 mô hình.
3. `results/visualizations/accuracy_curves_by_model.png` — Đường cong độ chính xác (Accuracy curves) 4 mô hình.
4. `results/visualizations/val_metrics_comparison_combined.png` — So sánh hội tụ Val Loss và Val Acc trên cùng một trục tọa độ.
5. `results/visualizations/parameters_breakdown.png` — Phân tích chi tiết số lượng tham số từng kiến trúc.
6. `results/visualizations/accuracy_vs_parameters.png` — Mối tương quan giữa độ chính xác và kích thước tham số (Bubble Chart).
7. `results/visualizations/training_time_comparison.png` — So sánh thời gian huấn luyện trên GPU Tesla T4.

