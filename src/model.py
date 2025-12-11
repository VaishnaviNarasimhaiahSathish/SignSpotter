import torch.nn as nn
from torchvision import models

# ---------------------------------------------------
# MobileNetV2 for Traffic Sign Classification
# ---------------------------------------------------

class TrafficSignMobileNet(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
