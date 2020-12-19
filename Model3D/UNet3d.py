import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        inter_channel = out_channel if in_channel > out_channel else out_channel // 2

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channel, inter_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channel//2),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(inter_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class OneForTwo(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OneForTwo, self).__init__()
        self.tend = BasicBlock(in_channel, out_channel)
        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.tend(x)
        return x, self.pool(x)


class cat_upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(cat_upsample, self).__init__()
        self.tend = BasicBlock(in_channel + in_channel//2, out_channel)
        self.sample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x1, x2):
        x2 = self.sample(x2)
        out = torch.cat((x1, x2), dim=1)
        out = self.tend(out)

        return out


class U_Net3d(nn.Module):
    def __init__(self, in_channel=3, num_classes=5):
        super(U_Net3d, self).__init__()
        self.block1 = OneForTwo(in_channel, 64)
        self.block2 = OneForTwo(64, 128)
        self.block3 = OneForTwo(128, 256)

        self.conv = BasicBlock(256, 512)

        self.block4 = cat_upsample(512, 256)
        self.block5 = cat_upsample(256, 128)
        self.block6 = cat_upsample(128, 64)

        self.classifier = nn.Conv3d(64, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1, x = self.block1(x)
        x2, x = self.block2(x)
        x3, x = self.block3(x)

        x = self.conv(x)

        x = self.block4(x3, x)
        x = self.block5(x2, x)
        x = self.block6(x1, x)

        out = self.classifier(x)
        return out

    