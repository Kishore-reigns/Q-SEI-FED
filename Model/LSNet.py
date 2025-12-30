# ===== Cell 5: LSNet =====
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)

    def forward(self, x):
        return self.dw(x)

class MCA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class MCAC(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = DWConv(c)
        self.mca = MCA(c)
        self.pw1 = nn.Conv2d(c, 4*c, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4*c, c, 1)

    def forward(self, x):
        y = self.dw(x)
        y = self.mca(y)
        y = self.act(self.pw1(y))
        y = self.pw2(y)
        return x + y

class LSNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        self.stage1 = nn.Sequential(*[MCAC(32) for _ in range(3)])
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.stage2 = nn.Sequential(*[MCAC(64) for _ in range(4)])
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.stage3 = nn.Sequential(*[MCAC(128) for _ in range(6)])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return F.normalize(self.fc(x))
