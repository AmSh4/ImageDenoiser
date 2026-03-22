import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import random
import re

# --- Configuration ---
IMAGE_SIZE = 256 #512
DATA_DIR = "./data"
PSEUDO_GT_DIR = os.path.join(DATA_DIR, "pseudo_ground_truth")

if not os.path.exists(PSEUDO_GT_DIR):
    os.makedirs(PSEUDO_GT_DIR)


def _extract_prefix(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    m = re.match(r'^(.*?)(\d+)$', name)
    if m:
        return m.group(1)
    return name


def _get_all_images(folder):
    """Return all image paths with common extensions."""
    exts = ['*.tif', '*.jpg', '*.jpeg', '*.png']
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    return sorted(files)


def build_pseudo_ground_truths_by_prefix(root_dir=DATA_DIR):
    noisy_path = os.path.join(root_dir, 'noisy_images')
    files = _get_all_images(noisy_path)
    if not files:
        raise FileNotFoundError(f"No noisy images found in {noisy_path}")

    groups = {}
    for f in files:
        prefix = _extract_prefix(f)
        groups.setdefault(prefix, []).append(f)

    created = []
    for prefix, flist in groups.items():
        out_path = os.path.join(PSEUDO_GT_DIR, f"{prefix}_pseudo_gt.tif")
        if os.path.exists(out_path):
            created.append(out_path)
            continue
        acc = None
        count = 0
        for f in flist:
            img = Image.open(f).convert("L")
            arr = np.array(img, dtype=np.float64)
            if acc is None:
                acc = arr
            else:
                acc += arr
            count += 1
        mean_img = (acc / count).round().astype(np.uint8)
        Image.fromarray(mean_img).save(out_path)
        created.append(out_path)
    return created


class MicroscopyClientDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.noisy_path = os.path.join(root_dir, 'noisy_images')
        self.noisy_image_files = _get_all_images(self.noisy_path)
        if not self.noisy_image_files:
            raise FileNotFoundError(f"No noisy images found in {self.noisy_path}")

        self.prefixes = [_extract_prefix(f) for f in self.noisy_image_files]

    def __len__(self):
        return len(self.noisy_image_files)

    def __getitem__(self, idx):
        noisy_image_path = self.noisy_image_files[idx]
        noisy_image = Image.open(noisy_image_path).convert("L")
        prefix = _extract_prefix(noisy_image_path)
        pseudo_gt_path = os.path.join(PSEUDO_GT_DIR, f"{prefix}_pseudo_gt.tif")
        if not os.path.exists(pseudo_gt_path):
            raise FileNotFoundError(f"Expected pseudo GT at {pseudo_gt_path}. Run build_pseudo_ground_truths_by_prefix().")
        gt_image = Image.open(pseudo_gt_path).convert("L")

        sample = {'noisy': noisy_image, 'ground_truth': gt_image}
        if self.transform:
            sample['noisy'] = self.transform(sample['noisy'])
            sample['ground_truth'] = self.transform(sample['ground_truth'])
        return sample


class Noise2NoiseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.noisy_path = os.path.join(root_dir, 'noisy_images')
        self.noisy_image_files = _get_all_images(self.noisy_path)
        self.num_images = len(self.noisy_image_files)
        if self.num_images < 2:
            raise ValueError("Noise2Noise requires at least two noisy images.")

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        rand_idx = random.randint(0, self.num_images - 1)
        while rand_idx == idx:
            rand_idx = random.randint(0, self.num_images - 1)

        path1 = self.noisy_image_files[idx]
        path2 = self.noisy_image_files[rand_idx]
        try:
            img1 = Image.open(path1).convert("L")
            img2 = Image.open(path2).convert("L")
        except Exception as e:
            return {'noisy_input': torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE),
                    'noisy_target': torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)}

        sample = {'noisy_input': img1, 'noisy_target': img2}
        if self.transform:
            sample['noisy_input'] = self.transform(sample['noisy_input'])
            sample['noisy_target'] = self.transform(sample['noisy_target'])
        return sample


def get_dataloader(batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(90),  # <-- Comment this line out
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_dataset = Noise2NoiseDataset(root_dir=DATA_DIR, transform=transform)
    print(f"Found {len(train_dataset)} noisy images for Noise2Noise training.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, None


def get_dataloader_for_evaluation(batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    full_dataset = MicroscopyClientDataset(root_dir=DATA_DIR, transform=transform)
    test_size = min(20, len(full_dataset))
    if len(full_dataset) - test_size <= 0:
        train_dataset, test_dataset = random_split(full_dataset, [0, len(full_dataset)])
    else:
        train_dataset, test_dataset = random_split(full_dataset, [len(full_dataset) - test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader
