import torch
import torch.nn as nn

x1 = torch.randn(2, 8, 4, 4)

x2 = x1.clone()
x2[1] = x2[1] * 100

bn = nn.BatchNorm2d(8)

y1_bn = bn(x1)
y2_bn = bn(x2)

print(torch.allclose(y1_bn[0], y2_bn[0]))

gn = nn.GroupNorm(
    num_groups=4,
    num_channels=8
)

y1_gn = gn(x1)
y2_gn = gn(x2)

print(torch.allclose(y1_gn[0], y2_gn[0]))