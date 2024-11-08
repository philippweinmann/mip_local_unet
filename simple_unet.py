import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleUNet, self).__init__()
        # Encoder
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.out = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        
        # Bottleneck
        b = self.bottleneck(x2)
        
        # Decoder
        u1 = self.up_conv1(b)
        u2 = self.up_conv2(u1)
        
        # Output
        out = self.out(u2)
        return torch.sigmoid(out)

