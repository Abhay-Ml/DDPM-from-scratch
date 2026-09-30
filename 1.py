
import torch
import torch.nn as nn

x = torch.randn(4, 32, 64, 64)

norm = nn.GroupNorm(
    num_groups=4,
    num_channels=32
)

y = norm(x)

print(x.shape)
print(y.shape)