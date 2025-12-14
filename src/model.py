import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                        stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, features=[64, 128, 256, 512]):
        super(ResUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        input_feat = features[0]
        for feature in features:
            self.encoder.append(ResidualBlock(input_feat, feature))
            input_feat = feature

        # Bottleneck
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        features = features[::-1]
        for feature in features:
            self.upconvs.append(nn.ConvTranspose2d(
                feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(ResidualBlock(feature * 2, feature))

        # Output
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        out = self.input_conv(x)

        for layer in self.encoder:
            out = layer(out)
            skip_connections.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.decoder)):
            out = self.upconvs[idx](out)
            skip = skip_connections[idx]
            if out.shape != skip.shape:
                out = F.interpolate(
                    out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            concat_skip = torch.cat((skip, out), dim=1)
            out = self.decoder[idx](concat_skip)

        out = self.final_conv(out)

        
        # --- GLOBAL RESIDUAL CONNECTION ---
        # We need to add the network's prediction (details) to the blurry input.
        # However, input 'x' has 6 channels, and 'out' has 2.
        # We extract the middle frame (Frame t, channels 2-3) from the input.
        base_frame = x[:, 2:4, :, :] 
        
        return base_frame + out

# --- Verification Code ---
if __name__ == "__main__":
    # Batch size 1, 6 channels (3 frames), 256x256
    input_tensor = torch.randn(1, 6, 256, 256)
    
    # Model configured for 2 output channels
    model = ResUNet(in_channels=6, out_channels=2)
    
    output = model(input_tensor)
    
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output.shape}") # Should be [1, 2, 256, 256]
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")