# %%
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

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
            # nn.BatchNorm3d(out_channels),
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

        # we're gonna need some padding here
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
def softdiceloss(predictions, targets, smooth: float = 0.001):
    batch_size = targets.shape[0]
    intersection = (predictions * targets).view(batch_size, -1).sum(-1)

    targets_area = targets.view(batch_size, -1).sum(-1)
    predictions_area = predictions.view(batch_size, -1).sum(-1)

    dice = (2. * intersection + smooth) / (predictions_area + targets_area + smooth)

    return 1 - dice.mean()

def dice_bce_loss(predictions, targets, weights = (1,1)):
    '''
    Combination between the bce loss and the soft dice loss. 
    The goal is to get the advantages
    from the soft dice loss without its potential instabilities.
    '''
    
    soft_dice_loss = softdiceloss(predictions, targets)

    bce_loss = nn.BCELoss()(predictions, targets)

    combination = weights[0] * soft_dice_loss + weights[1] * bce_loss

    return combination