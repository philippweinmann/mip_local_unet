# %%
import torch.nn as nn
import torch
import torch.nn.functional as F

import os

from typing import Optional

# Set the environment variable
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # maxpool3d is not implemented for MPS

# %%
# let's define the unet model here
class DoubleConv3D(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.conv_op = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
  
    def forward(self, x):
        return self.conv_op(x)
    
class DownSample(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.conv = DoubleConv3D(in_channels, out_channels)  
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  
  
    def forward(self, x):  
        down = self.conv(x)  
        p = self.pool(down)  
  
        return down, p
    
class UpSample(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm3d(in_channels//2)
        self.conv = DoubleConv3D(in_channels, out_channels)
  
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.batch_norm(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):  
        super().__init__()
        first_out_channels = 16
        self.down_convolution_1 = DownSample(in_channels, first_out_channels)  
        self.down_convolution_2 = DownSample(first_out_channels, first_out_channels * 2)  
        self.down_convolution_3 = DownSample(first_out_channels * 2, first_out_channels * 2 * 2)  
        self.down_convolution_4 = DownSample(first_out_channels * 2 * 2, first_out_channels * 2 * 2 * 2)
  
        self.bottle_neck = DoubleConv3D(first_out_channels * 2 * 2 * 2, first_out_channels * 2 * 2 * 2 * 2)
  
        self.up_convolution_1 = UpSample(first_out_channels * 2 * 2 * 2 * 2, first_out_channels * 2 * 2 * 2)  
        self.up_convolution_2 = UpSample(first_out_channels * 2 * 2 * 2, first_out_channels * 2 * 2)
        self.up_convolution_3 = UpSample(first_out_channels * 2 * 2, first_out_channels * 2)  
        self.up_convolution_4 = UpSample(first_out_channels * 2, first_out_channels)
  
        self.out = nn.Conv3d(in_channels=first_out_channels, out_channels=num_classes, kernel_size=1)
  
    def forward(self, x):  
        down_1, p1 = self.down_convolution_1(x)  
        down_2, p2 = self.down_convolution_2(p1)  
        down_3, p3 = self.down_convolution_3(p2)  
        down_4, p4 = self.down_convolution_4(p3)  
  
        b = self.bottle_neck(p4)  
  
        up_1 = self.up_convolution_1(b, down_4)  
        up_2 = self.up_convolution_2(up_1, down_3)  
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)
  
        out = self.out(up_4)

        # let's adapt the output for the loss
        out = torch.sigmoid(out)
        return out
    
# -------Loss-Functions----------
def softdiceloss(predictions, targets, smooth: float = 0.00001):
    batch_size = targets.shape[0]
    intersection = (predictions * targets).view(batch_size, -1).sum(-1)

    targets_area = targets.view(batch_size, -1).sum(-1)
    predictions_area = predictions.view(batch_size, -1).sum(-1)

    dice = (2. * intersection + smooth) / (predictions_area + targets_area + smooth)

    return 1 - dice.mean()

# reference implementation
class SoftDice(nn.Module):
    def __init__(self):
        """Dice coefficient."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None, smooth: float = 0):
        # Binarize prediction
        inputs = torch.where(inputs < self.thresh, 0, 1)
        batch_size = targets.shape[0]

        intersection = torch.logical_and(inputs, targets)
        intersection = intersection.view(batch_size, -1).sum(-1)
        targets_area = targets.view(batch_size, -1).sum(-1)
        inputs_area = inputs.view(batch_size, -1).sum(-1)
        dice = (2. * intersection + smooth) / (inputs_area + targets_area + smooth)

        if weights is not None:
            assert weights.shape == dice.shape, \
                f'"weights" must be in shape of "{dice.shape}"'
            return (dice * weights).sum()

        return dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self):
        """Dice loss + binary cross-entropy loss."""
        super().__init__()
        self.softdiceloss_fn = softdiceloss
        self.__name__ = 'DiceBCELoss'

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        batch_size = predictions.shape[0]

        bce = F.binary_cross_entropy(predictions, targets, reduction='mean')
        bce = bce.reshape(batch_size, -1).mean(-1)

        dice_loss = 1 - self.softdiceloss_fn(predictions, targets)
        dice_bce = bce + dice_loss
        return dice_bce