# %%
import torch.nn as nn
import torch

# %%
# let's define the unet model here
class DoubleConv(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.conv_op = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)  
        )
  
    def forward(self, x):
        return self.conv_op(x)
    
class DownSample(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        self.conv = DoubleConv(in_channels, out_channels)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
  
    def forward(self, x):  
        down = self.conv(x)  
        p = self.pool(down)  
  
        return down, p
    
class UpSample(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(in_channels//2)
        self.conv = DoubleConv(in_channels, out_channels)
  
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.batch_norm(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):  
        super().__init__()
        first_out_channels = 16
        self.down_convolution_1 = DownSample(in_channels, first_out_channels)  
        self.down_convolution_2 = DownSample(first_out_channels, first_out_channels * 2)  
        self.down_convolution_3 = DownSample(first_out_channels * 2, first_out_channels * 2 * 2)  
        self.down_convolution_4 = DownSample(first_out_channels * 2 * 2, first_out_channels * 2 * 2 * 2)
  
        self.bottle_neck = DoubleConv(first_out_channels * 2 * 2 * 2, first_out_channels * 2 * 2 * 2 * 2)
  
        self.up_convolution_1 = UpSample(first_out_channels * 2 * 2 * 2 * 2, first_out_channels * 2 * 2 * 2)  
        self.up_convolution_2 = UpSample(first_out_channels * 2 * 2 * 2, first_out_channels * 2 * 2)
        self.up_convolution_3 = UpSample(first_out_channels * 2 * 2, first_out_channels * 2)  
        self.up_convolution_4 = UpSample(first_out_channels * 2, first_out_channels)
  
        self.out = nn.Conv2d(in_channels=first_out_channels, out_channels=num_classes, kernel_size=1)
  
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
    
