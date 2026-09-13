"""
Module model_service: Quản lý việc khởi tạo, cấu hình, đóng băng/giải phóng trọng số (Freeze/Unfreeze),
thay thế Classification Head cho 4 kiến trúc pre-trained (ResNet-18, VGG-16, DenseNet-121, MobileNetV2),
và trực quan hóa chi tiết cấu trúc mạng bằng thư viện torchinfo (đáp ứng Step 3 của bài Lab).
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

from lab2_cv.config import SUPPORTED_MODELS, NUM_CLASSES


def freeze_features(model: nn.Module, freeze: bool = True) -> None:
    """
    Đóng băng hoặc mở khóa toàn bộ các tham số của mạng (trừ lớp phân loại cuối cùng).
    
    Cơ chế hoạt động:
    - Khi freeze=True: param.requires_grad = False, PyTorch sẽ không tính đạo hàm và không cập nhật
      trọng số các tầng tính năng (Feature Extractor) trong quá trình backpropagation.
    - Khi freeze=False: param.requires_grad = True, toàn bộ mạng được mở khóa để Fine-Tuning.
    
    Args:
        model (nn.Module): Mô hình neural network cần thao tác.
        freeze (bool): True để đóng băng, False để giải phóng trọng số.
    """
    # Nhận diện tên thuộc tính classifier tương ứng theo từng kiến trúc
    classifier_layer = None
    if hasattr(model, "fc"):
        classifier_layer = model.fc
    elif hasattr(model, "classifier"):
        classifier_layer = model.classifier

    for param in model.parameters():
        param.requires_grad = not freeze

    # Luôn đảm bảo lớp phân loại cuối cùng (Classifier Head) được huấn luyện
    if classifier_layer is not None:
        if isinstance(classifier_layer, nn.Sequential):
            # Với VGG hoặc MobileNet, lớp Linear cuối nằm trong nn.Sequential
            for p in classifier_layer[-1].parameters():
                p.requires_grad = True
        else:
            for p in classifier_layer.parameters():
                p.requires_grad = True


def unfreeze_last_layers(model: nn.Module, model_name: str) -> None:
    """
    Mở khóa (unfreeze) một vài tầng cuối cùng của mô hình để Fine-tuning,
    đáp ứng trực tiếp yêu cầu của Step 4 trong đề bài Lab:
    'Or you can unfreeze a few of the final layers to fine-tune the model.'
    
    Chi tiết các tầng được unfreeze theo từng kiến trúc:
    - ResNet-18: Mở khóa block cuối (model.layer4) và lớp phân loại (model.fc).
    - VGG-16: Mở khóa khối tích chập cuối (model.features[24:]) và classifier.
    - DenseNet-121: Mở khóa denseblock4, norm5 và model.classifier.
    - MobileNetV2: Mở khóa các inverted residual blocks cuối (model.features[14:]) và classifier.
    
    Args:
        model (nn.Module): Mô hình cần mở khóa một phần.
        model_name (str): Tên mô hình ('resnet18', 'vgg16', 'densenet121', 'mobilenet_v2').
    """
    model_name_lower = model_name.lower().strip()
    
    # 1. Trước tiên đóng băng toàn bộ mạng
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Mở khóa các tầng cuối tùy theo kiến trúc
    if model_name_lower == "resnet18":
        if hasattr(model, "layer4"):
            for param in model.layer4.parameters():
                param.requires_grad = True
        if hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True

    elif model_name_lower == "vgg16":
        if hasattr(model, "features"):
            for param in model.features[24:].parameters():
                param.requires_grad = True
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

    elif model_name_lower == "densenet121":
        if hasattr(model, "features"):
            if hasattr(model.features, "denseblock4"):
                for param in model.features.denseblock4.parameters():
                    param.requires_grad = True
            if hasattr(model.features, "norm5"):
                for param in model.features.norm5.parameters():
                    param.requires_grad = True
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

    elif model_name_lower == "mobilenet_v2":
        if hasattr(model, "features"):
            for param in model.features[14:].parameters():
                param.requires_grad = True
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True


def build_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    fine_tune_last_layers: bool = False,
) -> nn.Module:
    """
    Khởi tạo mô hình pre-trained từ torchvision, thay thế Classifier Head cho CIFAR-10 (10 classes),
    và cấu hình trạng thái đóng băng trọng số (Feature Extractor) hoặc Fine-tuning các tầng cuối.
    
    Chi tiết các kiến trúc được hỗ trợ:
    1. ResNet-18:
       - Classifier gốc: model.fc = Linear(in_features=512, out_features=1000)
       - Thay thế: model.fc = Linear(512, num_classes)
    2. VGG-16:
       - Classifier gốc: model.classifier[6] = Linear(in_features=4096, out_features=1000)
       - Thay thế: model.classifier[6] = Linear(4096, num_classes)
    3. DenseNet-121:
       - Classifier gốc: model.classifier = Linear(in_features=1024, out_features=1000)
       - Thay thế: model.classifier = Linear(1024, num_classes)
    4. MobileNetV2:
       - Classifier gốc: model.classifier[1] = Linear(in_features=1280, out_features=1000)
       - Thay thế: model.classifier[1] = Linear(1280, num_classes)
       
    Args:
        model_name (str): Tên mô hình trong SUPPORTED_MODELS.
        num_classes (int): Số lượng nhãn phân loại (mặc định CIFAR-10 là 10).
        freeze_backbone (bool): Đóng băng các tầng backbone (Feature Extractor) chỉ train lớp FC cuối.
        pretrained (bool): Nạp trọng số ImageNet pre-trained (True) hoặc khởi tạo ngẫu nhiên (False).
        fine_tune_last_layers (bool): Mở khóa một vài tầng cuối cùng để Fine-tuning (Step 4 trong Lab).
        
    Returns:
        nn.Module: Mô hình PyTorch đã được cấu hình hoàn chỉnh.
    """
    model_name_lower = model_name.lower().strip()
    if model_name_lower not in SUPPORTED_MODELS:
        raise ValueError(
            f"Mô hình '{model_name}' không được hỗ trợ! "
            f"Vui lòng chọn một trong các mô hình: {SUPPORTED_MODELS}"
        )

    # --------------------------------------------------------------------------
    # 1. ResNet-18 (Deep Residual Learning)
    # --------------------------------------------------------------------------
    if model_name_lower == "resnet18":
        if hasattr(models, "ResNet18_Weights"):
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(pretrained=pretrained)
        
        # Đóng băng toàn bộ tham số nếu có yêu cầu
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Thay thế Fully-Connected Layer cuối cùng (model.fc)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    # --------------------------------------------------------------------------
    # 2. VGG-16 (Very Deep Convolutional Networks)
    # --------------------------------------------------------------------------
    elif model_name_lower == "vgg16":
        if hasattr(models, "VGG16_Weights"):
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            model = models.vgg16(weights=weights)
        else:
            model = models.vgg16(pretrained=pretrained)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Trong VGG16, classifier là Sequential(Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear)
        # Lớp phân loại cuối cùng nằm ở index 6
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    # --------------------------------------------------------------------------
    # 3. DenseNet-121 (Densely Connected Convolutional Networks)
    # --------------------------------------------------------------------------
    elif model_name_lower == "densenet121":
        if hasattr(models, "DenseNet121_Weights"):
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
        else:
            model = models.densenet121(pretrained=pretrained)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Trong DenseNet-121, classifier là thuộc tính model.classifier
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    # --------------------------------------------------------------------------
    # 4. MobileNetV2 (Inverted Residuals and Linear Bottlenecks)
    # --------------------------------------------------------------------------
    elif model_name_lower == "mobilenet_v2":
        if hasattr(models, "MobileNet_V2_Weights"):
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v2(weights=weights)
        else:
            model = models.mobilenet_v2(pretrained=pretrained)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # Trong MobileNetV2, classifier là Sequential(Dropout(0.2), Linear(1280, 1000))
        # Lớp Linear nằm ở index 1
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # Nếu người dùng kích hoạt fine-tune một vài tầng cuối (Step 4 của Lab)
    if fine_tune_last_layers:
        unfreeze_last_layers(model, model_name=model_name_lower)

    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Thống kê số lượng tham số trong mô hình:
    - total_params: Tổng số lượng tham số
    - trainable_params: Số lượng tham số có requires_grad=True (được huấn luyện)
    - non_trainable_params: Số lượng tham số bị đóng băng (freezed)
    
    Args:
        model (nn.Module): Mô hình cần thống kê.
        
    Returns:
        Dict[str, int]: Dictionary chứa thống kê tham số.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }


def inspect_model(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (64, 3, 224, 224),
    device: str = "cpu",
    verbose: int = 1,
) -> Any:
    """
    Kiểm tra và in ra chi tiết cấu trúc mạng, kích thước tensor qua từng layer,
    số lượng tham số trainable / non-trainable bằng thư viện torchinfo.
    
    Hàm này đáp ứng trực tiếp yêu cầu của Step 3 trong đề bài Lab thực hành.
    
    Args:
        model (nn.Module): Mô hình cần kiểm tra.
        input_size (Tuple[int, int, int, int]): Kích thước tensor đầu vào giả lập [Batch, C, H, W].
        device (str): Thiết bị tính toán ('cpu' hoặc 'cuda').
        verbose (int): Mức độ chi tiết in ra (0: không in, 1: in chuẩn, 2: in đầy đủ).
        
    Returns:
        ModelStatistics: Đối tượng chứa đầy đủ thông tin thống kê của torchinfo.
    """
    model_summary = summary(
        model=model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
        device=device,
        verbose=verbose,
    )
    return model_summary
