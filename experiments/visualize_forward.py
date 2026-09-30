import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path

from PIL import Image
from torchvision import transforms

from src.ddpm.scheduler import NoiseScheduler
image_dir = Path(
    "data/raw/img_align_celeba/img_align_celeba"
)
image_path = next(image_dir.glob("*.jpg"))

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image)
image_tensor = image_tensor * 2 - 1
x_0 = image_tensor.unsqueeze(0)

print("x_0 shape:", x_0.shape)
noise = torch.randn_like(x_0)
t = torch.tensor([500])
scheduler = NoiseScheduler(
    num_timesteps=1000
)
print("alpha_bar[500]:", scheduler.alpha_bars[500])
print("sqrt_alpha_bar[500]:", scheduler.sqrt_alpha_bars[500])
print(
    "sqrt_one_minus_alpha_bar[500]:",
    scheduler.sqrt_one_minus_alpha_bars[500]
)
x_t = scheduler.add_noise(x_0, noise, t)
img_t = x_t.squeeze(0).permute(1, 2, 0)
img_t = (img_t + 1) / 2


plt.imshow(img_t)
plt.axis("off")
plt.show()