"""Build, adapt and inspect the four ImageNet pre-trained architectures."""

from typing import Any, Dict, Iterable, Tuple

import torch.nn as nn
from torchinfo import summary
from torchvision import models

from lab2_cv.config import MOBILENET_V4_TIMM_NAME, NUM_CLASSES, SUPPORTED_MODELS


def _classifier(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "resnet18":
        return model.fc
    if model_name == "vgg16":
        return model.classifier[6]
    if model_name == "densenet121":
        return model.classifier
    if model_name == "mobilenetv4_conv_small":
        return model.get_classifier()
    raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")


def _set_requires_grad(modules: Iterable[nn.Module], value: bool) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = value


def freeze_features(model: nn.Module, freeze: bool = True) -> None:
    """Compatibility helper: freeze/unfreeze all existing parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = not freeze


def unfreeze_last_layers(model: nn.Module, model_name: str) -> None:
    """Freeze the model, then enable the task head and final feature blocks."""
    model_name = model_name.lower().strip()
    freeze_features(model, True)
    modules: list[nn.Module] = [_classifier(model, model_name)]
    if model_name == "resnet18":
        modules.append(model.layer4)
    elif model_name == "vgg16":
        modules.append(model.features[24:])
    elif model_name == "densenet121":
        modules.extend([model.features.denseblock4, model.features.norm5])
    elif model_name == "mobilenetv4_conv_small":
        # timm MobileNetV4 exposes EfficientNet-style blocks and a final conv head.
        if hasattr(model, "blocks"):
            modules.append(model.blocks[-1])
        for attribute in ("conv_head", "bn2"):
            if hasattr(model, attribute):
                modules.append(getattr(model, attribute))
    _set_requires_grad(modules, True)


def build_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    fine_tune_last_layers: bool = False,
) -> nn.Module:
    """Load ImageNet weights, replace the classification head and set trainability."""
    model_name = model_name.lower().strip()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ: {SUPPORTED_MODELS}")

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            freeze_features(model, True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        if freeze_backbone:
            freeze_features(model, True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        if freeze_backbone:
            freeze_features(model, True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("MobileNetV4 cần thư viện timm: pip install timm") from exc
        model = timm.create_model(
            MOBILENET_V4_TIMM_NAME, pretrained=pretrained, num_classes=num_classes
        )
        if freeze_backbone:
            freeze_features(model, True)
            _set_requires_grad([model.get_classifier()], True)

    if not freeze_backbone:
        freeze_features(model, False)
    if fine_tune_last_layers:
        unfreeze_last_layers(model, model_name)
    return model


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Prevent frozen BatchNorm buffers from changing while the head is trained."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
    }


def inspect_model(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    verbose: int = 1,
) -> Any:
    """Print the model as required by the lab and return a compact torchinfo summary."""
    if verbose:
        print(model)
    return summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=18,
        row_settings=["var_names"],
        device=device,
        verbose=verbose,
    )
