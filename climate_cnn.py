import torch
import torch.nn as nn
from torchvision import models

class ClimateCNN(nn.Module):
    def __init__(self, num_classes=30):
        super(ClimateCNN, self).__init__()

        # Using pre-defined ResNet50 CNN architecture
        self.model = models.resnet50(weights='DEFAULT')

        # Modify first layer for 12 spectral bands input depth
        with torch.no_grad():
            old_weight = self.model.conv1.weight.data  # Shape: [64, 3, 7, 7]
            # Create new weight tensor for 12 channels
            new_weight = torch.cat([old_weight] * 4, dim=1)  # Shape: [64, 12, 7, 7]
            # Scale by 3/12 to keep the mean activation stable
            new_weight = new_weight * (3.0 / 12.0)

        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight.data = new_weight

        # Modify full-connected layer for number of output classes and add dropout
        num_features = int(self.model.fc.in_features)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
