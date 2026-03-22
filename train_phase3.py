# train_phase3.py
import os, sys, torch, csv
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.dataset import get_dataloader, build_pseudo_ground_truths_by_prefix, IMAGE_SIZE, DATA_DIR
from src.fc_mdm import FC_MDM
from src.model import UNet  # for compatibility if needed
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np   # 🔹 moved up (needed for refresh)
from tqdm import tqdm


from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image



# 🔹 Add config for auto pseudo ground truth refresh
UPDATE_PSEUDO_GT = True
PSEUDO_REFRESH_INTERVAL = 2  # refresh every 2 epochs (adjust if needed)


# def save_image_grid(noisy_batch, denoised_batch, epoch, out_dir="visual_results"):
#     os.makedirs(out_dir, exist_ok=True)
#     # Take first 4 samples (noisy above, denoised below)
#     noisy = (noisy_batch[:4].cpu() + 1) / 2
#     denoised = (denoised_batch[:4].cpu() + 1) / 2
#     combined = torch.cat([noisy, denoised], dim=0)  # 8 images total (4 noisy + 4 denoised)
#     grid = make_grid(combined, nrow=4)
#     img = transforms.ToPILImage()(grid)
#     img.save(os.path.join(out_dir, f"epoch_{epoch:02d}_grid.png"))


def save_image_grid_batch(noisy_batch, denoised_batch, epoch, out_dir="visual_results/grid"):
    """
    Save all images in the batch into grids of 8 (4 noisy above, 4 denoised below)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ensure batch dimension
    b = noisy_batch.size(0)
    n_per_grid = 4  # number of images per row/col for noisy+denoised

    for i in range(0, b, n_per_grid):
        noisy_slice = noisy_batch[i:i+n_per_grid].cpu()
        denoised_slice = denoised_batch[i:i+n_per_grid].cpu()

        # normalize to [0,1]
        noisy_slice = (noisy_slice + 1) / 2
        denoised_slice = (denoised_slice + 1) / 2

        # If less than n_per_grid, pad with black images
        if noisy_slice.size(0) < n_per_grid:
            pad = torch.zeros((n_per_grid - noisy_slice.size(0), *noisy_slice.shape[1:]))
            noisy_slice = torch.cat([noisy_slice, pad], dim=0)
            denoised_slice = torch.cat([denoised_slice, pad], dim=0)

        # Concatenate noisy on top, denoised below
        combined = torch.cat([noisy_slice, denoised_slice], dim=0)  # 8 images

        # Make grid: 4 per row (so 2 rows)
        grid = make_grid(combined, nrow=n_per_grid)
        img = transforms.ToPILImage()(grid)
        img.save(os.path.join(out_dir, f"epoch_{epoch:02d}_grid_{i//n_per_grid+1}.png"))



# 🔹 Define refresh helper ABOVE main()
def refresh_pseudo_ground_truth(model, dataloader, save_path="data/pseudo_ground_truth.npy"):
    """
    Re-generates pseudo ground truth using the current model.
    This makes the system self-improving without retraining from scratch.
    """
    print("\n🔁 Refreshing pseudo ground truth using current model...")
    model.eval()
    denoised_images = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Updating pseudo GT"):
            # 🔹 Fix: your dataloader returns dict with 'noisy_input'
            imgs = batch['noisy_input'].to(next(model.parameters()).device)
            # Use the full denoising loop to generate high-quality pseudo-GTs
            denoised = model.p_sample_loop_guided(
                noisy_init=imgs,
                timesteps=list(reversed(range(0, 500))) # Use more steps for better quality
            )
            denoised_images.append(denoised.cpu().numpy())

    # ❗️ ALSO, ADD THIS LINE AT THE END OF THE FUNCTION (around line 63)
    #    to ensure the model goes back to training mode after refreshing.

        print(f"✅ Updated pseudo ground truth saved at {save_path}")
        model.train() # <-- ADD THIS LINE

    all_imgs = np.concatenate(denoised_images, axis=0)
    np.save(save_path, all_imgs)
    print(f"✅ Updated pseudo ground truth saved at {save_path}")


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    LR = 2e-5
    NUM_EPOCHS = 8
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    OUT_PATH = os.path.join(MODEL_DIR, "phase3_fc_mdm_final.pth")

    print("Device:", DEVICE)
    print("Building pseudo ground truths...")
    try:
        build_pseudo_ground_truths_by_prefix(DATA_DIR)
    except Exception as e:
        print("Warning (pseudo GT creation):", e)

    loader, _ = get_dataloader(batch_size=BATCH_SIZE)
    print("Found dataloader with", len(loader), "batches.")

    model = FC_MDM(image_size=IMAGE_SIZE, num_timesteps=1000, device=DEVICE).to(DEVICE)

    baseline_path = os.path.join("models", "phase2_ddpm_noise2noise_256px.pth")
    if os.path.exists(baseline_path):
        try:
            sd = torch.load(baseline_path, map_location=DEVICE)
            model_state = model.state_dict()
            for k, v in sd.items():
                if k in model_state and sd[k].shape == model_state[k].shape:
                    model_state[k].copy_(sd[k])
            print("Loaded matching params from phase2 checkpoint (best-effort).")
        except Exception as e:
            print("Could not load baseline weights:", e)

    optimizer = Adam(model.parameters(), lr=LR)
    csv_path = os.path.join(MODEL_DIR, "phase3_training_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "avg_loss", "recon_loss", "spec_loss"])

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(NUM_EPOCHS):
        model.train()
        running = 0.0
        running_recon = 0.0
        running_spec = 0.0
        nb = 0
        pbar = tqdm(loader, desc=f"Phase3 Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch in pbar:
            inp = batch['noisy_input'].to(DEVICE)
            tgt = batch['noisy_target'].to(DEVICE)
            t = torch.randint(0, model.num_timesteps, (inp.shape[0],), device=DEVICE).long()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                total_loss, recon_loss, spec_loss = model.p_losses(tgt, inp, t, loss_type="l2", spectral_weight=10.0)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += total_loss.item()
            running_recon += recon_loss.item()
            running_spec += spec_loss.item()
            nb += 1
            pbar.set_postfix(loss=running / nb)

        avg = running / nb
        avg_recon = running_recon / nb
        avg_spec = running_spec / nb
        print(f"Epoch {epoch+1}: loss={avg:.6f} (recon={avg_recon:.6f}, spec={avg_spec:.6f})")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg, avg_recon, avg_spec])

        # save checkpoint each epoch
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"phase3_epoch{epoch+1}.pth"))


        try:
            with torch.no_grad():
                batch_vis = next(iter(loader))
                noisy_vis = batch_vis['noisy_input'].to(DEVICE)
                
                print("\nGenerating visualization grid...")
                model.eval() # Switch to evaluation mode for consistent output
                
                # Use the proper guided sampling loop to denoise the image
                denoised_vis = model.p_sample_loop_guided(
                    noisy_init=noisy_vis,
                    timesteps=list(reversed(range(0, 250))) # Use 250 steps for a quick but good result
                )
                
                model.train() # IMPORTANT: Switch back to training mode
                save_image_grid_batch(noisy_vis, denoised_vis, epoch+1)

        except Exception as e:
                print("Grid save skipped:", e)     


        # 🔹 Add self-refresh trigger here (safe, optional)
        if UPDATE_PSEUDO_GT and (epoch + 1) % PSEUDO_REFRESH_INTERVAL == 0:
            refresh_pseudo_ground_truth(model, loader)

    # final save
    torch.save(model.state_dict(), OUT_PATH)
    print("Saved final Phase-3 model to", OUT_PATH)


if __name__ == "__main__":
    main()
