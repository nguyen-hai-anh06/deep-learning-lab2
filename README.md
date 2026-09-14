# Practice 2 — Pre-trained Models on CIFAR-10

Dự án so sánh ResNet-18, VGG-16, DenseNet-121 và MobileNetV4 Conv Small bằng transfer learning. Ba model đầu dùng `torchvision`; MobileNetV4 dùng model `mobilenetv4_conv_small.e2400_r224_in1k` của `timm` với trọng số ImageNet-1K.

## Quy trình dữ liệu

- 45.000 ảnh train có augmentation.
- 5.000 ảnh validation để chọn checkpoint và hyperparameter.
- 10.000 ảnh test chỉ dùng trong bước đánh giá cuối.
- Seed mặc định là 42. File `split.json` trong mỗi run chứa checksum để kiểm tra các thành viên dùng cùng split.

## Chạy trên Google Colab

Mở `team_training_colab.ipynb`, bật GPU và chạy lần lượt các cell. Notebook lưu kết quả bền vững tại:

`practice2_cv_colab.ipynb` là notebook cũ chứa kết quả MobileNetV2; không dùng notebook đó cho đợt thực nghiệm mới.

```text
/content/drive/MyDrive/Lab2_Experiments/
  trainer_a/<run_id>/
  trainer_b/<run_id>/
```

Phân công mặc định:

- `trainer_a`: VGG-16 và MobileNetV4.
- `trainer_b`: ResNet-18 và DenseNet-121.

Mỗi người đổi biến `ROLE` trong notebook. File cấu hình tương ứng nằm trong `configs/`.
Nếu hai người dùng Drive riêng, sau khi train hãy chia sẻ nguyên thư mục `trainer_a`/`trainer_b` cho người quản lý. Người quản lý đặt cả hai thư mục dưới cùng một `Lab2_Experiments` trước khi tổng hợp; không đổi tên `run_id` hoặc file bên trong.

## Nội dung được lưu cho mỗi run

```text
<member_id>/<run_id>/
  config.json
  environment.json
  split.json
  architecture.txt
  console.log
  history.json
  history.csv
  summary.json
  checkpoints/
    last.pt
    best.pt
  tensorboard/
```

`last.pt` và lịch sử được ghi sau mỗi epoch. Nếu Colab ngắt, chạy lại cùng assignment; `run_assignment.py` tự resume từ `last.pt` và bỏ qua run đã hoàn thành. `last.pt` gồm model, optimizer, scheduler, AMP scaler và trạng thái random/data loader; `best.pt` là checkpoint gọn hơn dùng để đánh giá.

## Chạy một thí nghiệm

```bash
python train.py \
  --model mobilenetv4_conv_small \
  --strategy finetune_last \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.0001 \
  --optimizer adamw \
  --member-id trainer_a \
  --run-id mobilenetv4_finetune_last \
  --output-root /content/drive/MyDrive/Lab2_Experiments
```

Resume đúng run:

```bash
python train.py [các tham số giống lần đầu] --resume auto
```

Chạy toàn bộ phần được giao:

```bash
python run_assignment.py \
  --config configs/trainer_a.json \
  --output-root /content/drive/MyDrive/Lab2_Experiments \
  --data-dir /content/cifar10_data
```

## TensorBoard

```bash
tensorboard --logdir /content/drive/MyDrive/Lab2_Experiments
```

TensorBoard ghi train/validation loss, accuracy và learning rate theo epoch. Có thể thêm model graph khi chạy một model bằng `--log-graph`.

## Đánh giá cuối

Sau khi gom đủ thư mục của hai trainer, người quản lý tổng hợp các kết quả validation trước:

~~~bash
python collect_results.py \
  --experiments-root /content/drive/MyDrive/Lab2_Experiments
~~~

Sau khi kiểm tra bảng và đủ cả bốn model, chạy đánh giá test:

```bash
python run_evaluation.py \
  --experiments-root /content/drive/MyDrive/Lab2_Experiments \
  --output-dir /content/drive/MyDrive/Lab2_Experiments/final_evaluation \
  --data-dir /content/cifar10_data
```

Script chọn run tốt nhất của từng model dựa trên validation accuracy, kiểm tra checksum data split, rồi đánh giá một lần trên test set. Kết quả gồm accuracy, loss, macro precision/recall/F1, confusion matrix, tham số và thời gian train.

## Cài đặt local

```bash
pip install -r requirements.txt
pytest -q
```

Python 3.10+ được khuyến nghị.
