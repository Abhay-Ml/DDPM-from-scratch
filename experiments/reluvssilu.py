import torch
import torch.nn as nn

x = torch.tensor(
    [-2.0, -1.0, 0.0, 1.0, 2.0],
    requires_grad=True
)

relu = nn.ReLU()
silu = nn.SiLU()

y_relu = relu(x)
y_silu = silu(x)

print("ReLU:", y_relu)
print("SiLU:", y_silu)

x_relu = x.detach().clone().requires_grad_(True)
x_silu = x.detach().clone().requires_grad_(True)

y_relu = relu(x_relu)
y_silu = silu(x_silu)

y_relu.sum().backward()
y_silu.sum().backward()

print("ReLU gradients:", x_relu.grad)
print("SiLU gradients:", x_silu.grad)