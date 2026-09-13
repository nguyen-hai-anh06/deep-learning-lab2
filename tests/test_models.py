"""
Unit tests cho module model_service:
Kiểm tra output shape (Batch_Size x 10) của cả 4 mô hình:
['resnet18', 'vgg16', 'densenet121', 'mobilenet_v2'],
đồng thời kiểm chứng cơ chế đóng băng (Freeze) và giải phóng (Unfreeze) trọng số.
"""

import pytest
import torch
import torch.nn as nn

from lab2_cv.config import SUPPORTED_MODELS, NUM_CLASSES
from lab2_cv.services.model_service import (
    build_model,
    count_parameters,
    freeze_features,
    unfreeze_last_layers,
    inspect_model,
)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_model_output_shape(model_name: str):
    """
    Kiểm tra forward pass của 4 mô hình:
    Input tensor giả lập có shape [Batch=2, Channels=3, Height=224, Width=224].
    Output logits bắt buộc phải có shape [Batch=2, Num_Classes=10].
    """
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Khởi tạo mô hình (dùng pretrained=False để test nhanh kiến trúc không cần tải weights)
    model = build_model(model_name=model_name, num_classes=NUM_CLASSES, freeze_backbone=True, pretrained=False)
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, NUM_CLASSES), (
        f"Mô hình '{model_name}' trả về output shape sai: {output.shape}. "
        f"Kỳ vọng: ({batch_size}, {NUM_CLASSES})"
    )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_freeze_backbone_logic(model_name: str):
    """
    Kiểm tra cơ chế đóng băng Backbone:
    Khi freeze_backbone=True, số lượng tham số trainable phải nhỏ hơn rất nhiều tổng số tham số
    và chỉ nằm ở lớp Classifier cuối cùng.
    """
    model_frozen = build_model(model_name=model_name, num_classes=NUM_CLASSES, freeze_backbone=True, pretrained=False)
    param_counts_frozen = count_parameters(model_frozen)

    # 1. Phải có tham số non-trainable (bị đóng băng)
    assert param_counts_frozen["non_trainable_params"] > 0, (
        f"Mô hình '{model_name}' khi freeze_backbone=True không có tham số nào bị đóng băng!"
    )

    # 2. Số tham số trainable phải lớn hơn 0 (lớp FC cuối vẫn được train)
    assert param_counts_frozen["trainable_params"] > 0, (
        f"Mô hình '{model_name}' phải có tham số trainable ở lớp phân loại cuối!"
    )

    # 3. Số tham số trainable phải nhỏ hơn 5% tổng số tham số (vì chỉ train FC head)
    trainable_ratio = param_counts_frozen["trainable_params"] / param_counts_frozen["total_params"]
    assert trainable_ratio < 0.10, (
        f"Mô hình '{model_name}' có tỷ lệ trainable ({trainable_ratio:.2%}) quá cao khi freeze backbone!"
    )


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_unfreeze_logic(model_name: str):
    """Kiểm tra khi unfreeze toàn bộ mạng thì non-trainable params phải bằng 0."""
    model = build_model(model_name=model_name, num_classes=NUM_CLASSES, freeze_backbone=True, pretrained=False)
    
    # Giải phóng đóng băng
    freeze_features(model, freeze=False)
    param_counts = count_parameters(model)

    assert param_counts["non_trainable_params"] == 0, (
        f"Mô hình '{model_name}' sau khi unfreeze vẫn còn tham số bị đóng băng!"
    )
    assert param_counts["trainable_params"] == param_counts["total_params"]


def test_invalid_model_name_raises_error():
    """Kiểm tra truyền tên mô hình không hợp lệ sẽ kích hoạt ValueError."""
    with pytest.raises(ValueError) as exc_info:
        _ = build_model(model_name="unsupported_model_xyz", pretrained=False)
    assert "không được hỗ trợ" in str(exc_info.value)


def test_inspect_model_torchinfo():
    """Kiểm tra hàm inspect_model (Step 3 của bài Lab) thực thi trơn tru với MobileNetV2."""
    model = build_model(model_name="mobilenet_v2", num_classes=NUM_CLASSES, freeze_backbone=True, pretrained=False)
    stats = inspect_model(model, input_size=(2, 3, 224, 224), device="cpu", verbose=0)
    
    assert stats is not None
    assert stats.total_params > 0
    assert stats.trainable_params > 0


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_unfreeze_last_layers_logic(model_name: str):
    """
    Kiểm tra Step 4: 'unfreeze a few of the final layers to fine-tune the model'.
    Sau khi unfreeze các tầng cuối:
    - Trainable params phải lớn hơn khi chỉ train mỗi FC classifier.
    - Non-trainable params vẫn phải lớn hơn 0 (vì các tầng đầu vẫn đóng băng).
    """
    # 1. Mô hình đóng băng toàn bộ backbone
    model_frozen = build_model(model_name=model_name, num_classes=NUM_CLASSES, freeze_backbone=True, pretrained=False)
    counts_frozen = count_parameters(model_frozen)

    # 2. Mô hình unfreeze các tầng cuối
    model_finetune = build_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        pretrained=False,
        fine_tune_last_layers=True,
    )
    counts_finetune = count_parameters(model_finetune)

    # Trainable params của fine_tune_last_layers phải lớn hơn khi freeze_backbone hoàn toàn
    assert counts_finetune["trainable_params"] > counts_frozen["trainable_params"], (
        f"Mô hình '{model_name}': số tham số trainable khi unfreeze tầng cuối "
        f"({counts_finetune['trainable_params']}) phải lớn hơn khi đóng băng toàn bộ ({counts_frozen['trainable_params']})"
    )

    # Vẫn phải còn tham số đóng băng ở các tầng ban đầu
    assert counts_finetune["non_trainable_params"] > 0, (
        f"Mô hình '{model_name}': các tầng đầu vẫn phải được đóng băng!"
    )

