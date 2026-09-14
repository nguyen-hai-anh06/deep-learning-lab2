"""
Package lab2_cv: Core package cho bài thực hành Practice 2 - Transfer Learning trên CIFAR-10.
Cung cấp các module chuyên biệt: config, data_service, model_service, trainer_service, logger_service.
"""

import sys

# Đảm bảo console Windows luôn hỗ trợ tiếng Việt có dấu và ký tự đồ họa của torchinfo
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

__version__ = "2.0.0"

