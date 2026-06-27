import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UpConvDropBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.psi(F.relu(g1 + x1, inplace=True))
        return x * psi


class AttentionUNetGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.e2 = ConvBnReLU(base_ch,     base_ch * 2)
        self.e3 = ConvBnReLU(base_ch * 2, base_ch * 4)
        self.e4 = ConvBnReLU(base_ch * 4, base_ch * 8)
        self.e5 = ConvBnReLU(base_ch * 8, base_ch * 8)
        self.e6 = ConvBnReLU(base_ch * 8, base_ch * 8)
        self.e7 = ConvBnReLU(base_ch * 8, base_ch * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.d1 = UpConvDropBnReLU(base_ch * 8,     base_ch * 8, dropout=True)
        self.d2 = UpConvDropBnReLU(base_ch * 8 * 2, base_ch * 8, dropout=True)
        self.d3 = UpConvDropBnReLU(base_ch * 8 * 2, base_ch * 8, dropout=True)
        self.d4 = UpConvDropBnReLU(base_ch * 8 * 2, base_ch * 8)
        self.d5 = UpConvDropBnReLU(base_ch * 8 * 2, base_ch * 4)
        self.d6 = UpConvDropBnReLU(base_ch * 4 * 2, base_ch * 2)
        self.d7 = UpConvDropBnReLU(base_ch * 2 * 2, base_ch)

        self.att7 = AttentionGate(base_ch * 8, base_ch * 8, base_ch * 4)
        self.att6 = AttentionGate(base_ch * 8, base_ch * 8, base_ch * 4)
        self.att5 = AttentionGate(base_ch * 8, base_ch * 8, base_ch * 4)
        self.att4 = AttentionGate(base_ch * 8, base_ch * 8, base_ch * 4)
        self.att3 = AttentionGate(base_ch * 4, base_ch * 4, base_ch * 2)
        self.att2 = AttentionGate(base_ch * 2, base_ch * 2, base_ch)
        self.att1 = AttentionGate(base_ch,     base_ch,     base_ch // 2)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        bn = self.bottleneck(e7)

        d1 = self.d1(bn)
        d1 = torch.cat([d1, self.att7(d1, e7)], dim=1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, self.att6(d2, e6)], dim=1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, self.att5(d3, e5)], dim=1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, self.att4(d4, e4)], dim=1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, self.att3(d5, e3)], dim=1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, self.att2(d6, e2)], dim=1)
        d7 = self.d7(d6)
        d7 = torch.cat([d7, self.att1(d7, e1)], dim=1)

        return self.out(d7)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=2, base_ch=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch,     base_ch * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=1),
        )
    def forward(self, t1, t1ce):
        x = torch.cat([t1, t1ce], dim=1)
        return self.model(x)


def init_weights(net, gain=0.02):
    def _init(m):
        classname = m.__class__.__name__
        # Strictly check for primitive Convolutional layers
        if classname in ['Conv2d', 'ConvTranspose2d']:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        # Check for standard BatchNorm layers
        elif 'BatchNorm2d' in classname:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
    net.apply(_init)
