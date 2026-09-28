import torch
import torch.nn as nn
from torchvision.models import densenet161, resnet18

def get_densenet161(num_classes=5):
    """
    DenseNet161 모델을 불러와 출력 클래스 수(5개: A~E)에 맞게 최종 Classifier 레이어를 수정합니다.
    """
    model = densenet161(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def get_baseline_2dcnn(num_classes=5):
    """
    비교용 베이스라인 2D CNN 모델
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, num_classes)
    )

def get_baseline_resnet18(num_classes=5):
    """
    비교용 ResNet18 모델
    """
    res = resnet18(weights=None)
    res.fc = nn.Linear(res.fc.in_features, num_classes)
    return res