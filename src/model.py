import torch
import torch.nn as nn

# --- Generator (SRResNet) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))

class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, hidden_channels=64, num_res_blocks=16):
        super(Generator, self).__init__()
        
        # Initial Feature Extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Residual Trunk
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(hidden_channels))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels)
        )

        # Upsampling (two 2x blocks for 4x total)
        self.upsample = nn.Sequential(
            UpsampleBlock(hidden_channels, 2),
            UpsampleBlock(hidden_channels, 2)
        )

        # Final Reconstruction
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out = self.conv2(out)
        out = out + out1 # Global Skip Connection
        out = self.upsample(out)
        out = self.final_conv(out)
        return out

# --- Discriminator (VGG-Style) ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64):
        super(Discriminator, self).__init__()
        
        def conv_block(in_c, out_c, stride=1, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)]
            if bn: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.features = nn.Sequential(
            *conv_block(in_channels, hidden_channels, stride=1, bn=False),
            *conv_block(hidden_channels, hidden_channels, stride=2, bn=True),
            *conv_block(hidden_channels, hidden_channels*2, stride=1, bn=True),
            *conv_block(hidden_channels*2, hidden_channels*2, stride=2, bn=True),
            *conv_block(hidden_channels*2, hidden_channels*4, stride=1, bn=True),
            *conv_block(hidden_channels*4, hidden_channels*4, stride=2, bn=True),
            *conv_block(hidden_channels*4, hidden_channels*8, stride=1, bn=True),
            *conv_block(hidden_channels*8, hidden_channels*8, stride=2, bn=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels*8, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
            # No Sigmoid here because we use BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)