import torch

class NoiseScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    ):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, x_0, noise, t):
         sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
         sqrt_alpha_bar_t = sqrt_alpha_bar_t.reshape(-1, 1, 1, 1)

         sqrt_one_minus_alpha_bar_t = (
              self.sqrt_one_minus_alpha_bars[t]
    )
         sqrt_one_minus_alpha_bar_t = (
              sqrt_one_minus_alpha_bar_t.reshape(-1, 1, 1, 1)
    )   
         x_t= (sqrt_alpha_bar_t * x_0 +
               sqrt_one_minus_alpha_bar_t * noise
    )
         return x_t

if __name__ == "__main__":
    scheduler = NoiseScheduler()

    x_0 = torch.randn(4, 3, 64, 64)
    noise = torch.randn_like(x_0)

    t = torch.tensor([0, 250, 500, 999])

    x_t = scheduler.add_noise(x_0, noise, t)

    print("x_0 shape:", x_0.shape)
    print("noise shape:", noise.shape)
    print("t shape:", t.shape)
    print("x_t shape:", x_t.shape)