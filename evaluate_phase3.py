# evaluate.py
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import sys
import csv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model import UNet, BaselineDDPM
from src.dataset import get_dataloader_for_evaluation, build_pseudo_ground_truths_by_prefix, IMAGE_SIZE, DATA_DIR
from pytorch_msssim import ms_ssim

def tensor_to_pil(tensor):
    tensor = (tensor.cpu().clone().squeeze() + 1) / 2
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor.unsqueeze(0))

def evaluate_and_save(model, dataloader, device, out_dir="evaluation_results"):
    model.eval()
    results_dir = out_dir
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "mae", "psnr", "ssim", "ms_ssim"])

    total_mae = total_psnr = total_ssim = total_ms_ssim = 0.0
    num = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            noisy = batch['noisy'].to(device)
            gt = batch['ground_truth'].to(device)
            # conditional sampling from noisy image (shorter timesteps for speed)
            denoised = model.p_sample_loop_conditional(init_img=noisy, timesteps=list(reversed(range(800, 1000))))
            noisy_pil = tensor_to_pil(noisy)
            denoised_pil = tensor_to_pil(denoised)
            gt_pil = tensor_to_pil(gt)
            # save images
            denoised_pil.save(os.path.join(results_dir, f"{i}_denoised.png"))
            noisy_pil.save(os.path.join(results_dir, f"{i}_noisy.png"))
            gt_pil.save(os.path.join(results_dir, f"{i}_gt.png"))
            # metrics (convert to numpy uint8)
            den_np = np.array(denoised_pil)
            gt_np = np.array(gt_pil)
            mae = np.mean(np.abs(den_np.astype(float) - gt_np.astype(float)))
            p = psnr(gt_np, den_np, data_range=255)
            s = ssim(gt_np, den_np, data_range=255)
            # ms-ssim on tensors in [0,1]
            den_t = (denoised.clamp(-1, 1) + 1) / 2
            gt_t = (gt.clamp(-1, 1) + 1) / 2
            ms = ms_ssim(den_t, gt_t, data_range=1.0, size_average=True).item()

            total_mae += mae
            total_psnr += p
            total_ssim += s
            total_ms_ssim += ms
            num += 1

            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([i, mae, p, s, ms])

    # return averages
    return total_mae / num, total_psnr / num, total_ssim / num, total_ms_ssim / num

def main():
    MODEL_PATH = "models/phase2_ddpm_noise2noise_256px.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Step 1: Build pseudo ground truths if missing
    print("Building pseudo ground truths (if missing)...")
    created = build_pseudo_ground_truths_by_prefix(DATA_DIR)
    print(f"Created/Found {len(created)} pseudo GT images.")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run train.py first.")
        return

    unet_model = UNet(in_channels=1, out_channels=1)

    diffusion_model = BaselineDDPM(model=unet_model, image_size=IMAGE_SIZE, channels=1, num_timesteps=1000).to(device)
    #diffusion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded.")

    test_loader = get_dataloader_for_evaluation(batch_size=1)
    mae, p, s, ms = evaluate_and_save(diffusion_model, test_loader, device)
    print("\n--- Evaluation Complete ---")
    print(f"MAE: {mae:.2f}")
    print(f"PSNR: {p:.2f} dB")
    print(f"SSIM: {s:.4f}")
    print(f"MS-SSIM: {ms:.4f}")
    print("All results are in the 'evaluation_results' folder.")

if __name__ == "__main__":
    main()