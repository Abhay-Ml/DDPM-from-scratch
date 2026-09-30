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

scheduler = NoiseScheduler(
    num_timesteps=1000
)

timesteps = [0, 100, 250, 500, 750, 999]
noise = torch.randn_like(x_0)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for i, timestep in enumerate(timesteps):
    t = torch.tensor([timestep])
    x_t = scheduler.add_noise(x_0, noise, t)
    img_t = x_t.squeeze(0).permute(1, 2, 0)
    img_t = (img_t + 1) / 2
    img_t = img_t.clamp(0, 1)
    axes[i].imshow(img_t)
    axes[i].set_title(f"t={timestep}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()
