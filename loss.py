import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    def forward(self, pred, target):
        return self.loss(pred, target)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg.eval()
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:16]
        ).to(device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_3ch   = pred.repeat(1, 3, 1, 1)
        target_3ch = target.repeat(1, 3, 1, 1)
        feat_pred   = self.feature_extractor(pred_3ch)
        feat_target = self.feature_extractor(target_3ch)
        return F.l1_loss(feat_pred, feat_target)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('window', self._gaussian_window(window_size, sigma))

    @staticmethod
    def _gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)

    def _ssim(self, x, y):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        w = self.window.to(x.device)
        pad = self.window_size // 2
        mu_x  = F.conv2d(x, w, padding=pad)
        mu_y  = F.conv2d(y, w, padding=pad)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sig_x2 = F.conv2d(x * x, w, padding=pad) - mu_x2
        sig_y2 = F.conv2d(y * y, w, padding=pad) - mu_y2
        sig_xy = F.conv2d(x * y, w, padding=pad) - mu_xy
        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
        return (num / den).mean()

    def forward(self, pred, target):
        return 1.0 - self._ssim(pred, target)


class GeneratorLoss(nn.Module):
    def __init__(self, device, lambda_adv=1.0, lambda_l1=10.0,
                 lambda_perc=5.0, lambda_ssim=5.0):
        super().__init__()
        self.lambda_adv  = lambda_adv
        self.lambda_l1   = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.adv_loss  = AdversarialLoss()
        self.l1_loss   = L1Loss()
        self.perc_loss = PerceptualLoss(device)
        self.ssim_loss = SSIMLoss()

    def forward(self, fake_pred, fake_img, real_img):
        adv_loss  = self.adv_loss(fake_pred, is_real=True)
        l1_loss   = self.l1_loss(fake_img, real_img)
        perc_loss = self.perc_loss(fake_img, real_img)
        ssim_loss = self.ssim_loss(fake_img, real_img)
        total = (self.lambda_adv  * adv_loss
               + self.lambda_l1   * l1_loss
               + self.lambda_perc * perc_loss
               + self.lambda_ssim * ssim_loss)
        components = {
            'adv':  adv_loss.item(),
            'l1':   l1_loss.item(),
            'perc': perc_loss.item(),
            'ssim': ssim_loss.item(),
        }
        return total, components
