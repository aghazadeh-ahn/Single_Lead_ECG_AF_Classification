import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """Bلاک Residual کلاسیک، تطبیق‌یافته برای سیگنال 1D."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut اگر ابعاد تغییر کند
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)  # اتصال پرشی
        return out


class ResNet1D(nn.Module):
    """
    ResNet سبک برای ECG تک‌لیدی.
    ورودی: (batch, 1, seq_len)  ->  خروجی: (batch, num_classes)
    """
    def __init__(self, num_classes: int = 4,
                 channels=(32, 64, 128, 256),
                 blocks_per_stage=(2, 2, 2, 2),
                 dropout: float = 0.4):
        super().__init__()

        # Stem اولیه
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=15, stride=2,
                      padding=7, bias=False),   # کرنل بزرگ برای فرم ECG
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ۴ استیج با down-sampling
        layers, in_ch = [], channels[0]
        for i, (out_ch, n_blocks) in enumerate(zip(channels, blocks_per_stage)):
            for b in range(n_blocks):
                stride = 2 if (b == 0 and i > 0) else 1  # استیج اول بعد از stem
                layers.append(BasicBlock1D(in_ch, out_ch, stride))
                in_ch = out_ch
        self.res_layers = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x):
        # اگر ورودی (batch, seq_len) بود، کانال اضافه کن
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.res_layers(x)
        x = self.global_pool(x)
        return self.head(x)


if __name__ == "__main__":
    model = ResNet1D(num_classes=4)
    dummy = torch.randn(8, 1, 9000)  # مثلاً 30 ثانیه با 300Hz
    print(model(dummy).shape)  # -> torch.Size([8, 4])
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
