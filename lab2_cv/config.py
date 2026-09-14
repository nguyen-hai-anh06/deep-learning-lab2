"""
Module config: Chứa tất cả cấu hình hyperparameters, paths, và danh sách mô hình
dùng cho bài thực hành Practice 2 (CIFAR-10 Transfer Learning).
"""

import os
import torch

# ==============================================================================
# CẤU HÌNH PHẦN CỨNG & THIẾT BỊ (DEVICE)
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# CẤU HÌNH HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 64               # Kích thước batch phù hợp huấn luyện trên GPU T4/V100/A100 và CPU
LEARNING_RATE = 0.001          # Tốc độ học khởi tạo cho Optimizer (Adam/SGD)
NUM_EPOCHS = 10                # Số lượng epoch huấn luyện mặc định
NUM_CLASSES = 10               # Số lượng nhãn phân loại của CIFAR-10
VALIDATION_RATIO = 0.1         # 5,000 validation / 45,000 train
RANDOM_SEED = 42               # Dùng chung để mọi thành viên có cùng data split
MOMENTUM = 0.9                 # Momentum dùng khi cấu hình SGD optimizer
WEIGHT_DECAY = 1e-4            # L2 Regularization chống overfitting
NUM_WORKERS = 2                # Số tiến trình nạp dữ liệu đa luồng (multiprocessing)

# ==============================================================================
# DANH SÁCH 4 MÔ HÌNH PRE-TRAINED YÊU CẦU TRONG BÀI THỰC HÀNH
# ==============================================================================
SUPPORTED_MODELS = [
    "resnet18",       # He et al. - Residual Connections
    "vgg16",          # Simonyan & Zisserman - Deep Convolutional Networks
    "densenet121",    # Huang et al. - Densely Connected Convolutional Networks
    "mobilenetv4_conv_small",  # MobileNetV4 Conv Small từ timm, pre-trained ImageNet-1K
]

MOBILENET_V4_TIMM_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"

# ==============================================================================
# THỐNG SỐ XỬ LÝ ẢNH CHUẨN THEO PRE-TRAINED IMAGENET
# ==============================================================================
# Kích thước chuẩn đầu vào của các mạng pre-trained trên tập ImageNet-1k
IMAGE_SIZE = (224, 224)

# Thống kê Mean và Std của ImageNet để chuẩn hóa dữ liệu (RGB channels)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Tên 10 nhãn phân loại tương ứng trong bộ dữ liệu CIFAR-10
CIFAR10_CLASSES = [
    "airplane",     # Máy bay
    "automobile",   # Ô tô
    "bird",         # Chim
    "cat",          # Mèo
    "deer",         # Hươu
    "dog",          # Chó
    "frog",         # Ếch
    "horse",        # Ngựa
    "ship",         # Tàu thủy
    "truck",        # Xe tải
]

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC LƯU TRỮ (PATHS)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
