# hemorrhagic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HemorrhagicModel(nn.Module):
    """
    Model for classifying hemorrhagic conditions into 2 classes:
    Hemorrhagic, Normal.
    """
    def __init__(self):
        super(HemorrhagicModel, self).__init__()
        self.resnet = models.resnet18(weights=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer1.parameters():
            param.requires_grad = True
        for param in self.resnet.layer2.parameters():
            param.requires_grad = True
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(50688, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        resnet_features = self.resnet(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        combined_features = torch.cat((resnet_features, x), dim=1)
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)

        return x