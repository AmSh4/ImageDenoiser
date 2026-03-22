import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

# ============================================================
# Helper: Linear Beta Schedule
# ============================================================
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# ============================================================
# Exponential Moving Average Model
# ============================================================
class ModelEMA(object):
    def __init__(self, model, decay=0.9999):
        # Create an identical model for EMA tracking
        self.ema_model = type(model)()
        self.ema_model.load_state_dict(model.state_dict())
        self.decay = decay
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + msd[k].detach() * (1. - self.decay))

# ============================================================
# Sinusoidal Time Embeddings
# ============================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ============================================================
# Residual Block (used inside UNet)
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU()
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)

# ============================================================
# UNet Architecture (Base Model for DDPM)
# ============================================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Downsampling
        self.down1 = ResidualBlock(in_channels, 64, time_emb_dim)
        self.down2 = ResidualBlock(64, 128, time_emb_dim)
        self.down3 = ResidualBlock(128, 256, time_emb_dim)

        # Middle Block
        self.mid = ResidualBlock(256, 256, time_emb_dim)

        # Upsampling
        self.up1 = ResidualBlock(256, 128, time_emb_dim)
        self.up2 = ResidualBlock(128, 64, time_emb_dim)
        self.up3 = ResidualBlock(64, out_channels, time_emb_dim)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        t = self.time_mlp(t)
        x1 = self.down1(x, t)
        x2 = self.down2(self.maxpool(x1), t)
        x3 = self.down3(self.maxpool(x2), t)
        x_mid = self.mid(x3, t)
        x = self.up1(self.upsample(x_mid), t)
        x = self.up2(self.upsample(x), t)
        x = self.up3(self.upsample(x), t)
        return x

# ============================================================
# DDPM (Denoising Diffusion Probabilistic Model)
# ============================================================
class BaselineDDPM(nn.Module):
    def __init__(self, model, image_size, channels, num_timesteps):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.num_timesteps = num_timesteps

        betas = linear_beta_schedule(num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    # Forward Diffusion (add noise)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # Training loss
    def p_losses(self, x_start, t, loss_type="l2"):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    # Reverse sampling loop
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            alpha_t = 1. - self.betas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            predicted_noise = self.model(img, t)
            term1 = (1.0 / torch.sqrt(alpha_t)).view(-1, 1, 1, 1)
            term2 = (self.betas[t] / torch.sqrt(1. - alpha_t_cumprod)).view(-1, 1, 1, 1)
            img = term1 * (img - term2 * predicted_noise)
            if i > 0:
                noise = torch.randn_like(img)
                img += torch.sqrt(self.betas[t]).view(-1, 1, 1, 1) * noise
        return img

    # Conditional sampling from a given noisy image
    @torch.no_grad()
    def p_sample_loop_conditional(self, init_img, timesteps=None):
        device = self.betas.device
        b = init_img.shape[0]
        img = init_img.to(device)
        if timesteps is None:
            timesteps = list(reversed(range(0, self.num_timesteps)))
        for i in tqdm(timesteps, desc='Conditional Sampling', total=len(timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            alpha_t = 1. - self.betas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            predicted_noise = self.model(img, t)
        return img
