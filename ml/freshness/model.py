"""
Module: ml/freshness/model.py
Purpose: Lightweight Transfer Learning Model for Produce Freshness Classification
Architecture: MobileNetV3-Small (default) or ResNet18
"""

import torch
import torch.nn as nn
from torchvision import models

def build_freshness_model(architecture: str = "mobilenet_v3_small", pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Builds a transfer learning classifier.
    num_classes = 2 (0: Fresh, 1: Stale)
    """
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        
        # Replace classifier head
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model
