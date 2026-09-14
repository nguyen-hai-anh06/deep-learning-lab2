# Practice 2 — Khung báo cáo kết quả

> Chưa điền số liệu trước khi hoàn thành các run. Dùng `final_evaluation/test_results.csv` và TensorBoard làm nguồn duy nhất khi cập nhật báo cáo.

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

## 4. Kiến trúc và tham số

Chèn bảng tổng hợp từ `architecture.txt` và `config.json`:

| Model | Classification layer | Total params | Trainable freeze | Trainable fine-tune |
|---|---|---:|---:|---:|
| ResNet-18 | `fc` | TBD | TBD | TBD |
| VGG-16 | `classifier[6]` | TBD | TBD | TBD |
| DenseNet-121 | `classifier` | TBD | TBD | TBD |
| MobileNetV4 | `classifier` qua `reset_classifier/get_classifier` | TBD | TBD | TBD |

## 5. Kết quả validation và test

Điền từ file do `run_evaluation.py` sinh ra:

| Model | Strategy được chọn | Val accuracy | Test accuracy | Test loss | Macro F1 | Train time |
|---|---|---:|---:|---:|---:|---:|
| ResNet-18 | TBD | TBD | TBD | TBD | TBD | TBD |
| VGG-16 | TBD | TBD | TBD | TBD | TBD | TBD |
| DenseNet-121 | TBD | TBD | TBD | TBD | TBD | TBD |
| MobileNetV4 | TBD | TBD | TBD | TBD | TBD | TBD |

## 6. Nhận xét cần viết sau thực nghiệm

- So sánh độ chính xác với số tham số và thời gian train.
- Dùng đường train/validation trên TensorBoard để nhận xét hội tụ và overfitting.
- So sánh freeze với fine-tune cho từng model.
- Dùng confusion matrix để nêu các cặp lớp dễ nhầm.
- Ghi rõ GPU khi so sánh thời gian; chỉ so sánh latency được đo trên cùng thiết bị.
