import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Binary(nn.Module):
    def __init__(self, in_channels):
        super(CNN_Binary, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        
        x = self.fc(x)
        return x
