import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2

        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32)
            * math.log(10000)
            / (half_dim - 1)
        )

        self.register_buffer("freqs", freqs)

    def forward(self, t):
        t = t.float()

        angles = t[:, None] * self.freqs

        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)

        embedding = torch.cat([sin_part, cos_part], dim=-1)

        return embedding


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )

        self.act = nn.SiLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )

        self.time_proj = nn.Linear(
            time_dim,
            out_channels
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )

    def forward(self, x, time_emb):

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        time = self.time_proj(time_emb)
        time = time[:, :, None, None]

        h = h + time

        h = self.conv2(h)
        h = self.norm2(h)

        residual = self.shortcut(x)

        h = h + residual
        h = self.act(h)

        return h

block = ResidualBlock(
    in_channels=64,
    out_channels=128,
    time_dim=128
)

x = torch.randn(4, 64, 32, 32)
time_emb = torch.randn(4, 128)

output = block(x, time_emb)

print("Input:", x.shape)
print("Time:", time_emb.shape)
print("Output:", output.shape)