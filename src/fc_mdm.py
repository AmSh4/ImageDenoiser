# src/fc_mdm.py
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from tqdm import tqdm
from .model import UNet  # reuse existing UNet
import math

# Simple conv block used for fusion/refinement
def conv3x3(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.SiLU()
    )

class DualResolutionUNet(nn.Module):
    """
    Two U-Nets in parallel: full-res and low-res. Their feature outputs are fused.
    For simplicity we use lightweight UNet instances (reuse your UNet class).
    """
    def __init__(self, full_cfg, low_cfg, fuse_channels=64):
        super().__init__()
        # full resolution UNet (heavier)
        self.full_unet = UNet(in_channels=1, out_channels=1)
        # low resolution UNet (lighter) - we instantiate another UNet but it will operate on downsampled input
        self.low_unet = UNet(in_channels=1, out_channels=1)
        # fusion layers: combine predicted noise maps from both streams and refine
        self.fuse = nn.Sequential(
            conv3x3(2, fuse_channels),
            conv3x3(fuse_channels, fuse_channels),
            nn.Conv2d(fuse_channels, 1, kernel_size=1)
        )

    def forward(self, x_full, x_low, t):
        """
        x_full: full-resolution noisy image tensor (B,1,H,W)
        x_low: low-resolution noisy image (B,1,H_low,W_low) - should be a downsampled version
        t: timestep embeddings (torch.long tensor of shape (B,))
        """
        # Predict noise map from each stream independently
        pred_full = self.full_unet(x_full, t)
        pred_low = self.low_unet(x_low, t)

        # make sure both are same spatial size
        pred_full = F.interpolate(pred_full, size=x_full.shape[-2:], mode='bilinear', align_corners=False)
        pred_low_up = F.interpolate(pred_low, size=x_full.shape[-2:], mode='bilinear', align_corners=False)

        fused = torch.cat([pred_full, pred_low_up], dim=1)
        out = self.fuse(fused)
        return out  # predicted noise

class Refiner(nn.Module):
    """A lightweight refinement UNet to self-correct predictions (embedded loop)."""
    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, t=None):
        return self.refine(x)

class FC_MDM(nn.Module):
    """
    Frequency-Constrained Multi-scale Diffusion Model (FC-MDM).
    Wraps DualResolutionUNet + Refiner + loss/guidance helpers.
    """
    def __init__(self, image_size=256, num_timesteps=1000, device='cpu'):
        super().__init__()
        self.image_size = image_size
        self.num_timesteps = num_timesteps
        self.device = device

        # Dual-resolution UNet
        self.dual_unet = DualResolutionUNet(full_cfg=None, low_cfg=None, fuse_channels=64)
        # Refiner (internal refinement loop)
        self.refiner = Refiner()

        # Create simple DDPM betas / buffers (re-use same schedule)
        betas = self.linear_beta_schedule(num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, noisy_input, t, loss_type="l2", spectral_weight=1.0):
        """
        Training loss combining noise prediction L2 and spectral consistency penalty.
        - x_start: target image (noisy target in Noise2Noise) in normalized [-1,1]
        - noisy_input: paired noisy input (same shape)
        - t: timesteps
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # build low-res version by downsampling
        x_low = F.interpolate(x_noisy, scale_factor=0.5, mode='bilinear', align_corners=False)
        predicted_noise = self.dual_unet(x_noisy, x_low, t)
        # optional internal refinement: reconstruct and re-refine prediction
        # compute initial denoised estimation: x0_pred = (x_noisy - sqrt_one_minus_alphas * predicted_noise) / sqrt_alphas
        sqrt_a_t = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_om_a_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        x0_pred = (x_noisy - sqrt_om_a_t * predicted_noise) / sqrt_a_t
        # run refiner on x0_pred (embedded loop)
        refined = self.refiner(x0_pred)
        # combine: small residual from refiner
        x0_refined = x0_pred + 0.1 * refined

        # compute noise target for loss (we used noise sampling)
        if loss_type == 'l2':
            loss_recon = F.mse_loss(predicted_noise, noise)
        else:
            loss_recon = F.l1_loss(predicted_noise, noise)

        # spectral consistency penalty: discourage added HF power not supported by noisy_input
        spec_loss = self.spectral_consistency_loss(x0_refined, noisy_input)

        total_loss = loss_recon + spectral_weight * spec_loss
        return total_loss, loss_recon.detach(), spec_loss.detach()

    def spectral_consistency_loss(self, denoised, noisy_input, hf_band_ratio=0.25, eps=1e-8):
        """
        Compute penalty when denoised image introduces more high-frequency power than noisy_input.
        Strategy:
         - compute magnitude spectra of denoised and noisy_input
         - compute HF mask (frequencies outside central low-pass radius)
         - compute excess HF power = max(0, mag_denoised - mag_noisy_scaled)
         - return mean excess power
        """
        # convert to FFT complex (B, H, W)
        # operate on single channel images
        b, c, h, w = denoised.shape
        assert c == 1
        # compute 2D FFT magnitudes (shifted)
        def mag(x):
            X = fft.rfft2(x.squeeze(1), norm='ortho')
            M = torch.abs(X)
            return M

        mag_den = mag(denoised)
        mag_noi = mag(noisy_input)

        # estimate noise floor as median of noisy magnitude
        floor = torch.median(mag_noi.view(b, -1), dim=1)[0].view(b,1,1)
        # scale noisy mag slightly up to allow some amplification
        mag_noi_scaled = mag_noi + 0.5 * floor

        # Create high-frequency mask in rfft domain by frequency radius threshold
        freqs_y = torch.fft.fftfreq(h, device=denoised.device)[:h]  # length h
        freqs_x = torch.fft.rfftfreq(w, device=denoised.device)    # length w//2+1
        yy = freqs_y.unsqueeze(1).abs().unsqueeze(0)  # (1,h,1)
        xx = freqs_x.unsqueeze(0).abs().unsqueeze(0)  # (1,1,wf)
        # compute radial frequency normalized
        radius = torch.sqrt(yy**2 + xx**2)
        cutoff = float(hf_band_ratio) * torch.max(radius)
        hf_mask = (radius >= cutoff).float()  # shape (1,h,wf)
        hf_mask = hf_mask.to(denoised.device)

        # compute excess HF power where denoised magnitude exceeds scaled noisy magnitude
        excess = F.relu(mag_den - mag_noi_scaled) * hf_mask
        loss = excess.mean()
        return loss

    @torch.no_grad()
    def p_sample_loop_guided(self, noisy_init, timesteps=None, guidance_strength=0.8, hf_band_ratio=0.25):
        """
        Reverse sampling conditioned on noisy input and guided iteratively by spectral constraints.
        noisy_init: noisy input image normalized [-1,1], shape (B,1,H,W)
        timesteps: list of timesteps (descending). If None, use full schedule.
        guidance_strength: how strongly to pull frequencies back to input where input lacks power.
        """
        device = noisy_init.device
        b = noisy_init.shape[0]
        img = noisy_init.clone().to(device)

        if timesteps is None:
            timesteps = list(reversed(range(0, self.num_timesteps)))

        for i in tqdm(timesteps, desc='Conditional Sampling', total=len(timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            # downsample for low res stream
            x_low = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
            # predict noise
            predicted_noise = self.dual_unet(img, x_low, t)
            # step predictor (simple ancestral step as in baseline)
            alpha_t = 1. - self.betas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            term1 = (1.0 / torch.sqrt(alpha_t)).view(-1,1,1,1)
            term2 = (self.betas[t] / torch.sqrt(1. - alpha_t_cumprod)).view(-1,1,1,1)
            img = term1 * (img - term2 * predicted_noise)

            # small stochasticity
            if i > 0:
                noise = torch.randn_like(img) * 0.01
                img = img + torch.sqrt(self.betas[t]).view(-1,1,1,1) * noise

            # iterative spectral guidance: attenuate HF components that appear unsupported by input
            img = self.apply_spectral_guidance(img, noisy_init, strength=guidance_strength, hf_band_ratio=hf_band_ratio)

            # internal refinement: re-run refiner on current denoised estimate every few steps (cheap)
            if (i % 50) == 0:
                # reconstruct x0 estimate
                sqrt_a_t = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
                sqrt_om_a_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
                x0_est = (img - sqrt_om_a_t * predicted_noise) / (sqrt_a_t + 1e-12)
                refined = self.refiner(x0_est)
                img = img + 0.05 * refined  # small correction

        return img

    def apply_spectral_guidance(self, img, noisy_input, strength=0.8, hf_band_ratio=0.25):
        """
        Pull high-frequency magnitudes of img toward those of noisy_input where the noisy_input
        lacks coherent power. This reduces hallucinated HF textures.
        Implementation: operate in rfft2 domain and blend magnitudes.
        """
        # shapes
        b, c, h, w = img.shape
        # compute rfft2 (complex) for both
        X_img = fft.rfft2(img.squeeze(1), norm='ortho')   # (B, H, Wf)
        X_noi = fft.rfft2(noisy_input.squeeze(1), norm='ortho')
        mag_img = torch.abs(X_img)
        mag_noi = torch.abs(X_noi)

        # HF mask (same as spectral_loss)
        freqs_y = torch.fft.fftfreq(h, device=img.device)[:h]
        freqs_x = torch.fft.rfftfreq(w, device=img.device)
        yy = freqs_y.unsqueeze(1).abs().unsqueeze(0)
        xx = freqs_x.unsqueeze(0).abs().unsqueeze(0)
        radius = torch.sqrt(yy**2 + xx**2)
        cutoff = float(hf_band_ratio) * torch.max(radius)
        hf_mask = (radius >= cutoff).float().to(img.device)  # (1,h,wf)

        # create allowed magnitude at HF: noisy mag + small floor
        floor = torch.median(mag_noi.view(b, -1), dim=1)[0].view(b,1,1)
        allowed = mag_noi + 0.5 * floor

        # where mag_img > allowed, we push it toward allowed:
        excess = F.relu(mag_img - allowed) * hf_mask
        # compute blending factor per-frequency
        # small factor to nudge magnitudes; stronger strength -> more aggressive replacement
        blend = (1.0 - torch.exp(-strength * (excess / (allowed + 1e-8)))) * hf_mask

        # new magnitude = mag_img * (1 - blend) + allowed * blend
        new_mag = mag_img * (1.0 - blend) + allowed * blend

        # rebuild complex spectrum: keep phase of img, set magnitude to new_mag
        phase = X_img / (mag_img + 1e-12)
        X_new = new_mag * phase
        # inverse rfft2
        x_new = fft.irfft2(X_new, s=(h, w), norm='ortho').unsqueeze(1).to(img.device)
        return x_new
