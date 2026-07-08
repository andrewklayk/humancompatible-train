"""Model factories, copied faithfully from benchmark/utils.py."""
import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def create_mlp(input_shape, latent_size1=64, latent_size2=32):
    return Sequential(
        torch.nn.Linear(input_shape, latent_size1),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size1, latent_size2),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size2, 1),
    )


def create_conv(num_classes=10):
    return ConvNet(num_classes=num_classes)


def _resnet18(num_classes, cifar_stem):
    """torchvision resnet18. ``cifar_stem=False`` keeps the ImageNet stem (7x7 stride-2
    conv + maxpool), matching the original benchmark's ``create_resnet``.
    ``cifar_stem=True`` replaces conv1 with a 3x3 stride-1 conv and drops the maxpool so
    32x32 CIFAR inputs aren't downsampled away before the residual stages (the standard
    CIFAR adaptation)."""
    import torchvision  # lazy import: keep models.py usable without torchvision installed
    model = torchvision.models.resnet18(num_classes=num_classes)
    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def create_resnet(num_classes=10):
    """Faithful torchvision resnet18 (ImageNet stem) -- matches the original benchmark."""
    return _resnet18(num_classes, cifar_stem=False)


def create_resnet_cifar(num_classes=10):
    """resnet18 with a CIFAR-adapted stem (3x3 conv1, no maxpool), for 32x32 inputs."""
    return _resnet18(num_classes, cifar_stem=True)
