import torch
import math

embedding_dim = 8
half_dim = embedding_dim // 2

freqs = torch.exp(
    -torch.arange(half_dim)
    * math.log(10000)
    / (half_dim - 1)
)

t = torch.tensor([10, 500, 999], dtype=torch.float32)

result = t[:, None] * freqs

sin_part = result.sin()
cos_part = result.cos()

embedding = torch.cat([sin_part, cos_part], dim=-1)

print("freqs:", freqs)
print("result shape:", result.shape)
print("embedding:", embedding)
print("embedding shape:", embedding.shape)