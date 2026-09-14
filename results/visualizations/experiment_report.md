# Báo Cáo Trực Quan Hóa Thực Nghiệm — CIFAR-10 Transfer Learning

> Báo cáo được tự động khởi tạo từ các kết quả huấn luyện thực tế trong `Lab2_Experiments`.

## 1. Bảng Tổng Hợp Kết Quả Thực Nghiệm

| Mô hình | Chiến lược | Thành viên | Tổng tham số | Tham số Trainable | Val Accuracy | Best Epoch | Final Train Loss | Final Val Loss | Thời gian train |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 | Freeze | trainer_b | 11,181,642 | 5,130 | **86.46%** | 5 | 0.4692 | 0.4078 | 17.5 phút |
| ResNet-18 | Fine-tune Last | trainer_b | 11,181,642 | 8,398,858 | **95.22%** | 9 | 0.0312 | 0.1583 | 18.1 phút |
| DenseNet-121 | Freeze | trainer_b | 6,964,106 | 10,250 | **88.62%** | 7 | 0.3823 | 0.3669 | 21.7 phút |
| DenseNet-121 | Fine-tune Last | trainer_b | 6,964,106 | 2,170,378 | **94.40%** | 8 | 0.0708 | 0.1631 | 24.4 phút |
| VGG-16 | Freeze | trainer_a | 134,301,514 | 40,970 | **82.20%** | 7 | 0.8525 | 0.5372 | 26.9 phút |
| VGG-16 | Fine-tune Last | trainer_a | 134,301,514 | 7,120,394 | **93.86%** | 9 | 0.0681 | 0.2015 | 29.3 phút |
| MobileNetV4 (Conv Small) | Freeze | trainer_a | 2,505,834 | 12,810 | **87.96%** | 9 | 0.4056 | 0.3808 | 18.0 phút |
| MobileNetV4 (Conv Small) | Fine-tune Last | trainer_a | 2,505,834 | 1,366,410 | **92.32%** | 10 | 0.1299 | 0.2604 | 18.3 phút |

---

## 2. Các Biểu Đồ Trực Quan Hóa

### A. So Sánh Hiệu Quả Giữa Freeze và Fine-tune (Accuracy Gain)
Biểu đồ thể hiện sự vượt trội rõ rệt khi giải phóng các tầng đặc trưng cuối cùng (Fine-tune Last Layers) so với việc chỉ huấn luyện tầng phân loại (Freeze Backbone).

![So sánh Freeze vs Fine-tune](strategy_gain_comparison.png)

### B. Quá Trình Hội Tụ Loss Qua Các Epoch (Loss Curves)
Theo dõi hiện tượng giảm hàm mất mát trên cả tập huấn luyện và validation, làm nổi bật bước chuyển tại Epoch 5 do cơ chế StepLR ($\gamma=0.1$).

![Đường cong hàm mất mát](loss_curves_by_model.png)

### C. Diễn Biến Độ Chính Xác (Accuracy Curves)
Biểu diễn mức độ cải thiện độ chính xác qua 10 epoch của 4 mô hình.

![Đường cong độ chính xác](accuracy_curves_by_model.png)

### D. So Sánh Đồng Thời 4 Kiến Trúc (Fine-tune Last)
Đánh giá trực tiếp tốc độ hội tụ và độ chính xác của ResNet-18, DenseNet-121, VGG-16 và MobileNetV4 trên cùng một hệ trục tọa độ.

![So sánh đồng thời 4 mô hình](val_metrics_comparison_combined.png)

### E. Phân Tích Số Lượng Tham Số (Parameters Breakdown)
Khảo sát sự chênh lệch lớn về kích thước mô hình (VGG-16 134.3M tham số vs MobileNetV4 chỉ 2.5M tham số).

![Phân tích tham số](parameters_breakdown.png)

### F. Đánh Đổi Giữa Độ Chính Xác và Kích Thước Mô Hình (Bubble Chart)
Trực quan hóa tính kinh tế / hiệu năng của từng kiến trúc. Bong bóng lớn hơn biểu thị thời gian huấn luyện lâu hơn.

![Độ chính xác vs Số tham số](accuracy_vs_parameters.png)

### G. So Sánh Thời Gian Huấn Luyện (Training Time)
Tổng thời gian tiêu tốn cho 10 epoch trên phần cứng Nvidia Tesla T4.

![Thời gian huấn luyện](training_time_comparison.png)

---

## 3. Nhận Xét & Phân Tích Khoa Học

1. **Hiệu quả vượt trội của Fine-tune Last:**
   - Cả 4 kiến trúc đều ghi nhận mức tăng trưởng độ chính xác đáng kể khi chuyển từ Freeze sang Fine-tune:
     - **VGG-16**: tăng mạnh nhất **+11.66%** (từ 82.20% lên 93.86%). Nguyên nhân do VGG-16 chỉ có các tầng Conv tuần tự, việc unfreeze block conv cuối (block 5) giúp các bộ lọc học thích ứng rất tốt với đặc trưng kích thước nhỏ của CIFAR-10.
     - **ResNet-18**: tăng **+8.76%** (từ 86.46% lên 95.22%), đạt vị trí **quán quân về độ chính xác** trong toàn bộ thí nghiệm.
     - **DenseNet-121**: tăng **+5.78%** (từ 88.62% lên 94.40%), đứng thứ hai về độ chính xác.
     - **MobileNetV4**: tăng **+4.36%** (từ 87.96% lên 92.32%).

2. **Độ kinh tế của kiến trúc (Architectural Efficiency):**
   - **MobileNetV4 Conv Small** chỉ có **2.51M tham số** (kích thước ~9.56 MB), nhẹ hơn VGG-16 tới **53.6 lần**, nhưng đạt độ chính xác **92.32%** (chỉ kém VGG-16 1.54%). Thời gian train chỉ mất 18.3 phút.
   - **ResNet-18** (11.18M tham số) là mô hình cân bằng hoàn hảo nhất: thời gian train nhanh nhất (18.1 phút), độ chính xác cao nhất (95.22%). Nhờ cấu trúc Residual Skip Connections triệt tiêu hiện tượng vanishing gradient.
   - **VGG-16** (134.3M tham số) cồng kềnh nhất, thời gian train lâu nhất (29.3 phút) do khối lượng phép tính lớn và số lượng tham số khổng lồ.

3. **Tác động của Learning Rate Scheduler (StepLR):**
   - Tại Epoch 5, khi Learning Rate giảm 10 lần (từ 1e-4 xuống 1e-5 ở Fine-tune, hoặc 1e-3 xuống 1e-4 ở Freeze), đồ thị Loss của cả 4 mô hình đều giảm một bậc dốc rõ rệt, và Accuracy tăng vọt thêm 1-2% rồi ổn định dần, chứng minh hiệu quả hội tụ tối ưu của StepLR.
