import os
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from timm.utils import ModelEmaV2
from unet import UNET


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------- Configuration (edit here) ----------------
CHECKPOINT = 'stl10_checkpoint.pt'   # path to checkpoint
SCHEDULER = 'cosine'                 # 'linear' or 'cosine'
MODE = 'recon_sweep'                 # 'recon', 'recon_sweep', 'sample', 'collage'
T_START = 700                        # for recon mode
T_LIST = [700, 500, 300, 200]        # for recon_sweep
CFG_SCALE = 2.0                     # classifier-free guidance scale at sampling
ETA = 0.0                            # 0.0 -> DDIM (deterministic), 1.0 -> DDPM-like
LABEL = -1                           # 0-9 class, -1 random, 10 null/unconditional
IDX = 0                              # dataset index for recon modes
SPLIT = 'train'                      # 'train' or 'test'
OUTPUT_DIR = 'intermiate imagees'    # where to save outputs
USE_EMA = True                       # use EMA weights for inference


# --- Schedulers (match training) ---
class DDPM_beta_t_linear_scheduler(nn.Module):
    def __init__(self, num_steps: int = 1000):
        super().__init__()
        beta_t = torch.linspace(1e-4, 0.02, num_steps, requires_grad=False)
        alpha = 1 - beta_t
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer('beta_t', beta_t)
        self.register_buffer('alpha_t', alpha_bar)


class DDPM_beta_t_cosine_scheduler(nn.Module):
    def __init__(self, num_steps: int = 1000, s: float = 0.008):
        super().__init__()
        self.T = num_steps
        t = torch.arange(0, self.T + 1, dtype=torch.float32)
        alpha_bar = torch.cos(((t / self.T + s) / (1 + s)) * math.pi / 2) ** 2
        betas = torch.clamp(1 - (alpha_bar[1:] / alpha_bar[:-1]), min=1e-4, max=0.999)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('beta_t', betas)
        self.register_buffer('alpha_t', alpha_cumprod)


# --- Helpers ---
def _tensor_to_image(img_tensor: torch.Tensor):
    img = img_tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    img = torch.clamp((img + 1) / 2, 0, 1)
    return img.permute(1, 2, 0).numpy()


def _cfg_predict_noise(model: nn.Module, z: torch.Tensor, t_tensor: torch.Tensor, labels: torch.Tensor, guidance_scale: float):
    null_labels = torch.full_like(labels, 10)
    e_uncond = model(z, t_tensor, null_labels)
    e_cond = model(z, t_tensor, labels)
    return e_uncond + guidance_scale * (e_cond - e_uncond)


def reverse_sample(model: nn.Module,
                   scheduler: nn.Module,
                   z: torch.Tensor,
                   start_t: int,
                   labels: torch.Tensor,
                   cfg_scale: float,
                   eta: float) -> torch.Tensor:
    for t in reversed(range(1, start_t + 1)):
        t_tensor = torch.tensor([t], device=z.device)
        predicted_noise = _cfg_predict_noise(model, z, t_tensor, labels, cfg_scale)

        alpha_bar_t = scheduler.alpha_t[t]
        beta_t = scheduler.beta_t[t]
        alpha_t = 1.0 - beta_t
        alpha_bar_prev = scheduler.alpha_t[t-1] if t > 0 else torch.tensor(1.0, device=z.device)

        z = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)
        if t > 1 and eta > 0.0:
            noise = torch.randn_like(z)
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            z = z + eta * torch.sqrt(posterior_var) * noise

    # final step t=0
    t_tensor = torch.tensor([0], device=z.device)
    predicted_noise = _cfg_predict_noise(model, z, t_tensor, labels, cfg_scale)
    alpha_bar_0 = scheduler.alpha_t[0]
    beta_0 = scheduler.beta_t[0]
    alpha_0 = 1.0 - beta_0
    x0 = (1.0 / torch.sqrt(alpha_0)) * (z - (beta_0 / torch.sqrt(1.0 - alpha_bar_0)) * predicted_noise)
    return x0


def save_side_by_side(images: List[torch.Tensor], titles: List[str], out_path: str):
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(2 * cols, 2))
    if cols == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        if isinstance(img, torch.Tensor):
            img = _tensor_to_image(img)
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


def main():
    # Transforms as in training
    TRANS = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Data for reconstruction modes
    dataset = torchvision.datasets.STL10(root='stl10_data', split=SPLIT, download=True, transform=TRANS)

    # Model
    model = UNET(time_steps=1000, input_channels=3, output_channels=3, label_embedding=True, use_film=False, dropout_prob=0.0)
    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['weights'])
    ema = ModelEmaV2(model, decay=0.999)
    ema.load_state_dict(ckpt['ema'])
    model_eval = ema.module if USE_EMA else model
    model_eval.eval()

    # Scheduler
    if SCHEDULER == 'linear':
        sched = DDPM_beta_t_linear_scheduler(1000).to(device)
    else:
        sched = DDPM_beta_t_cosine_scheduler(1000).to(device)

    # Labels for sampling
    if LABEL == 10:
        labels = torch.tensor([10], device=device)
    elif LABEL == -1:
        labels = torch.randint(0, 10, (1,), device=device)
    else:
        labels = torch.tensor([int(LABEL)], device=device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if MODE == 'recon':
        x, label = dataset[IDX]
        x = x.unsqueeze(0).to(device)
        if CFG_SCALE == 0.0 or LABEL == 10:
            labels = torch.tensor([10], device=device)
        # make x_t
        t0 = max(1, min(999, int(T_START)))
        a_bar = sched.alpha_t[t0].to(device).view(1, 1, 1, 1)
        e = torch.randn_like(x)
        x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * e
        x0 = reverse_sample(model_eval, sched, x_t.clone(), t0, labels, CFG_SCALE, ETA)
        save_side_by_side([x, x_t, x0], ["orig", f"x_t (t={t0})", "recon"], os.path.join(OUTPUT_DIR, f'recon_t{t0}.png'))

    elif MODE == 'recon_sweep':
        # Build sweep panel: orig + recon at list of t
        x, label = dataset[IDX]
        x = x.unsqueeze(0).to(device)
        if CFG_SCALE == 0.0 or LABEL == 10:
            labels = torch.tensor([10], device=device)
        imgs = [_tensor_to_image(x)]
        titles = ['orig']
        for t0 in T_LIST:
            t0 = max(1, min(999, int(t0)))
            a_bar = sched.alpha_t[t0].to(device).view(1, 1, 1, 1)
            e = torch.randn_like(x)
            x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * e
            x0 = reverse_sample(model_eval, sched, x_t.clone(), t0, labels, CFG_SCALE, ETA)
            imgs.append(_tensor_to_image(torch.clamp(x0, -1, 1)))
            titles.append(f'recon t={t0}')
        save_side_by_side(imgs, titles, os.path.join(OUTPUT_DIR, 'recon_sweep.png'))

    elif MODE == 'sample':
        # Pure sampling from noise
        z = torch.randn(1, 3, 96, 96, device=device)
        x0 = reverse_sample(model_eval, sched, z, 999, labels, CFG_SCALE, ETA)
        plt.figure(figsize=(3, 3))
        plt.imshow(_tensor_to_image(torch.clamp(x0, -1, 1)))
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, 'sample.png'), bbox_inches='tight', dpi=150)
        plt.close()

    elif MODE == 'collage':
        # Denoising collage from noise
        timesteps = [999, 700, 550, 400, 300, 200, 100, 50, 15, 0]
        z = torch.randn(1, 3, 96, 96, device=device)
        snapshots = {}
        for t in reversed(range(1, 1000)):
            t_tensor = torch.tensor([t], device=device)
            predicted_noise = _cfg_predict_noise(model_eval, z, t_tensor, labels, CFG_SCALE)
            alpha_bar_t = sched.alpha_t[t]
            beta_t = sched.beta_t[t]
            alpha_t = 1.0 - beta_t
            alpha_bar_prev = sched.alpha_t[t-1] if t > 0 else torch.tensor(1.0, device=device)
            z = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)
            if t > 1 and ETA > 0.0:
                noise = torch.randn_like(z)
                posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                z = z + ETA * torch.sqrt(posterior_var) * noise
            if t in timesteps:
                snapshots[t] = torch.clamp(z, -1, 1)
        # final step
        t_tensor = torch.tensor([0], device=device)
        predicted_noise = _cfg_predict_noise(model_eval, z, t_tensor, labels, CFG_SCALE)
        alpha_bar_0 = sched.alpha_t[0]
        beta_0 = sched.beta_t[0]
        alpha_0 = 1.0 - beta_0
        x0 = (1.0 / torch.sqrt(alpha_0)) * (z - (beta_0 / torch.sqrt(1.0 - alpha_bar_0)) * predicted_noise)
        snapshots[0] = torch.clamp(x0, -1, 1)

        ordered = [snapshots[t] for t in timesteps if t in snapshots]
        titles = [f't={t}' for t in timesteps if t in snapshots]
        save_side_by_side(ordered, titles, os.path.join(OUTPUT_DIR, 'collage.png'))


if __name__ == '__main__':
    main()


