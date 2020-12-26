import torch
import torch.nn as nn

operation_canditates = {
    '00': lambda C, stride: Zero(stride),
    '01': lambda C, stride: Inception(C, C, 3, stride, 1),
    '10': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    '11': lambda C, stride: ResSepConv(C, C, 3, stride, 1),
}


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias):
        super(Conv2d, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.GroupNorm(6, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)


class Inception(nn.Module):
    def __init__(self, C_in, C_out, ksize, stride, padding):
        super(Inception, self).__init__()
        self.branch1 = Conv2d(C_in, C_out // 2, kernel_size=ksize, stride=stride,
                              padding=padding, bias=False)
        self.branch2 = Conv2d(C_in, C_out // 2, kernel_size=ksize, stride=stride,
                              padding=padding, bias=False)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat([x1, x2], dim=1)
        return out


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, ksize, stride, padding):
        super(SepConv, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=ksize, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(6, C_in, affine=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=ksize, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(6, C_out, affine=False)
        )

    def forward(self, x):
        return self.features(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ResSepConv, self).__init__()
        self.conv = SepConv(C_in, C_out, kernel_size, stride, padding)
        self.res = Identity() if stride == 1 else FactorizedReduce(C_in, C_out)

    def forward(self, x):
        return sum([self.conv(x), self.res(x)])


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(6, C_out, affine=False)
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.GroupNorm(6, C_out, affine=False)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
