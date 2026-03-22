# MicroClear: Advanced Noise Removal Using Diffusion Models
## Project Overview

This project implements a high-quality noise removal system for images using a custom diffusion-based model (FC-MDM). The system takes noisy images, predicts their clean version, and supports evaluation using pseudo ground truth images. The goal is to create a model that generalizes well, even on unseen data, outperforming standard denoising models.

---
The project uses Retrieval-Augmented Generation (RAG)-style ideas for pseudo ground truth creation in image denoising, which is rare in conventional diffusion denoising pipelines.

---
## Key Features

1) Fully Convolutional Multi-Scale Diffusion Model (FC-MDM)

- Custom UNet-inspired architecture for noise removal.

- Works on multi-scale image patches for better generalization.

- File: src/fc_mdm.py

2) Noise2Noise Training Approach

- Trains the model without clean ground truth images, reducing dataset preparation effort.

- File: src/model.py

4) Pseudo Ground Truth Generation

- Automatically creates pseudo clean images using ensemble denoising for evaluation.

- File: src/dataset.py

5) Phase-Based Training Pipeline

- Supports modular training phases (Phase 1, 2, 3) for iterative model improvement.

- Files: train_phase3.py (Phase 3 training)

6) Visualization of Results

- Generates grids of denoised images for qualitative analysis.

- File: evaluate.py

---
## Project Structure
            Project_FC_MDM/
            │
            ├─ src/
            │   ├─ dataset.py           # Data loading, preprocessing, pseudo GT creation
            │   ├─ fc_mdm.py            # Main FC-MDM model architecture
            │   ├─ fc_mdm_model.py      # Advanced FC-MDM extensions (optional)
            │   └─ model.py             # UNet + baseline DDPM models
            │
            ├─ train_phase3.py          # Script to train Phase 3 model
            ├─ evaluate.py              # Evaluate model on pseudo ground truth
            ├─ evaluate_phase3.py       # Phase 3 evaluation (optional)
            ├─ models/
            │   └─ phase2_ddpm_noise2noise_256px.pth  # Pretrained weights
            └─ data/
                ├─ noisy_images/        # Input noisy images
                └─ pseudo_ground_truth/ # Generated pseudo clean images

## Tech Stack

- Python 3.11+: Main language.

- PyTorch 2.x: For building and training diffusion models.

- NumPy & SciPy: Numerical processing.

- Pillow: Image reading, processing, and saving.

- Matplotlib: Result visualization.

- tqdm: Progress bars for training.

- scikit-learn & pandas: Optional metrics and data handling.

---
## How to Run

1) Install dependencies

- pip install -r requirements.txt


2) Prepare your dataset

- Place noisy images in data/noisy_images/.

- Pseudo ground truth will be automatically generated in data/pseudo_ground_truth/ (or you can provide precomputed ones).

3) Train the model (Phase 3 example)

- python train_phase3.py --data_dir data/noisy_images/ --epochs 50 --batch_size 16


4) Evaluate the model

- python evaluate_phase3.py --model_path models/


5) Visualize results

- Evaluation script saves grids of original vs denoised images automatically.

## Detailed File Explanations

- dataset.py	- Loads images, creates pseudo ground truth, handles preprocessing.
- fc_mdm.py 	- Defines the fully convolutional multi-scale diffusion model.
- fc_mdm_model.py	- Optional extended model architecture for research/Phase 3.
- model.py	- UNet and baseline DDPM models; supports Noise2Noise training.
- train_phase3.py	- Training script for Phase 3; customizable hyperparameters.
- evaluate_phase3.py	- Evaluation script specifically for Phase 3 outputs.
## Unique Selling Points

1) Pseudo Ground Truth Pipeline: Reduces dependency on fully clean datasets.

2) Multi-Scale FC-MDM: Works better on varied noise patterns than typical UNet DDPM.

3) Phase-Based Modular Training: Easy to extend and improve iteratively.

4) Visual Grids for Evaluation: Instant visual feedback on denoising quality.

**These features are rare in standard diffusion-based denoising pipelines, and make this project highly client-ready for research or production use.**

