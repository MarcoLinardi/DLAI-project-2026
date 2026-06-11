"""ResNet-20 for CIFAR-10/100 (He et al., 2015).

Reference: "Deep Residual Learning for Image Recognition", Sec. 4.2.
The CIFAR variant uses a 3x3 stem conv (not 7x7 stride 2 like ImageNet),
three stages of {16, 32, 64} channels with 3 BasicBlocks each, and
global average pooling before the classifier. Total: ~270K params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block with two 3x3 convolutions and an identity shortcut.

    y = ReLU( BN(conv2( ReLU(BN(conv1(x))) )) + shortcut(x) )
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, option: str = "B") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            if option == "A":
                # Zero-padded identity shortcut (paper-faithful, no extra params).
                pad = planes * self.expansion - in_planes
                self.shortcut = _ZeroPadShortcut(stride=stride, pad_channels=pad)
            elif option == "B":
                # Projection shortcut: 1x1 conv. Tiny param overhead, easier to merge.
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * self.expansion),
                )
            else:
                raise ValueError(f"Unknown shortcut option: {option!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class _ZeroPadShortcut(nn.Module):
    """Option-A shortcut: subsample spatially then pad new channels with zeros."""

    def __init__(self, stride: int, pad_channels: int) -> None:
        super().__init__()
        self.stride = stride
        self.pad_channels = pad_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.pad_channels > 0:
            zeros = x.new_zeros(x.size(0), self.pad_channels, x.size(2), x.size(3))
            x = torch.cat([x, zeros], dim=1)
        return x


class ResNetCIFAR(nn.Module):
    """ResNet for 32x32 inputs. Depth = 6n + 2 (n blocks per stage)."""

    def __init__(self, num_blocks_per_stage: int = 3, num_classes: int = 10, option: str = "B") -> None:
        super().__init__()
        self.in_planes = 16
        self.option = option

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, num_blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(32, num_blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(64, num_blocks_per_stage, stride=2)

        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        # First block may downsample; subsequent blocks keep spatial size and channels.
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, stride=s, option=self.option))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet20(num_classes: int = 10, option: str = "B") -> ResNetCIFAR:
    return ResNetCIFAR(num_blocks_per_stage=3, num_classes=num_classes, option=option)


if __name__ == "__main__":
    model = resnet20()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"params: {n_params:,}  |  output: {tuple(y.shape)}")
