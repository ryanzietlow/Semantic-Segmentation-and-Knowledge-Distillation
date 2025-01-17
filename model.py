import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactSegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super(CompactSegmentationModel, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder with batch normalization
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn_up1 = nn.BatchNorm2d(64)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.bn_up2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final classification with proper initialization
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        # Store input size
        input_size = x.size()[2:]

        # Encoding path with residual connections
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))

        # Decoding path with careful skip connections
        x = self.bn_up1(self.upconv1(x3))
        x = x + x2  # Skip connection
        x = self.conv4(x)

        x = self.bn_up2(self.upconv2(x))
        x = x + x1  # Skip connection
        x = self.conv5(x)

        # Final 1x1 convolution
        x = self.final_conv(x)

        # Ensure output size matches input size with align_corners=True
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        if return_features:
            return x, x3
        return x