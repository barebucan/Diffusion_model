import os
import math
import argparse
from typing import List

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from timm.utils import ModelEmaV2
from unet import UNET


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Scheduler definitions (match training: 500 steps) ---
class DDPM_beta_t_linear_scheduler(nn.Module):
    def __init__(self, num_steps: int = 500):
        super().__init__()
        beta_t = torch.linspace(1e-4, 0.02, num_steps, requires_grad=False)
        alpha = 1 - beta_t
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer('beta_t', beta_t)
        self.register_buffer('alpha_t', alpha_bar)


class DDPM_beta_t_cosine_scheduler(nn.Module):
    def __init__(self, num_steps: int = 500, s: float = 0.008):
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
def _tensor_to_image(x: torch.Tensor):
    x = x.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    x = torch.clamp((x + 1) / 2, 0, 1)
    return x.permute(1, 2, 0).numpy()


def _cfg_predict_eps_from_v(model: nn.Module, scheduler, z: torch.Tensor, t_tensor: torch.Tensor, labels: torch.Tensor, guidance_scale: float):
    null_labels = torch.full_like(labels, 10)
    v_uncond = model(z, t_tensor, null_labels)
    v_cond = model(z, t_tensor, labels)
    v_hat = v_uncond + guidance_scale * (v_cond - v_uncond)
    alpha_bar_t = scheduler.alpha_t[t_tensor]
    alpha_t_sqrt = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
    sigma_t = torch.sqrt(1.0 - alpha_bar_t).view(-1, 1, 1, 1)
    eps_hat = sigma_t * z + alpha_t_sqrt * v_hat
    return eps_hat


@torch.no_grad()
def _sample_from_noise(model: nn.Module, scheduler, label: int, guidance_scale: float, eta: float, num_steps: int, seed: int):
    torch.manual_seed(seed)
    z = torch.randn(1, 3, 96, 96, device=device)
    labels = torch.tensor([label], device=device, dtype=torch.long)
    for t in reversed(range(1, num_steps)):
        t_tensor = torch.tensor([t], device=device)
        predicted_noise = _cfg_predict_eps_from_v(model, scheduler, z, t_tensor, labels, guidance_scale)
        alpha_bar_t = scheduler.alpha_t[t]
        beta_t = scheduler.beta_t[t]
        alpha_t = 1.0 - beta_t
        alpha_bar_prev = scheduler.alpha_t[t-1] if t > 0 else torch.tensor(1.0, device=device)
        z = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)
        if t > 1 and eta > 0.0:
            noise = torch.randn_like(z)
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            z = z + eta * torch.sqrt(posterior_var) * noise
    # t=0
    t_tensor = torch.tensor([0], device=device)
    predicted_noise = _cfg_predict_eps_from_v(model, scheduler, z, t_tensor, labels, guidance_scale)
    alpha_bar_0 = scheduler.alpha_t[0]
    beta_0 = scheduler.beta_t[0]
    alpha_0 = 1.0 - beta_0
    x0 = (1.0 / torch.sqrt(alpha_0)) * (z - (beta_0 / torch.sqrt(1.0 - alpha_bar_0)) * predicted_noise)
    return torch.clamp(x0, -1, 1)


def _save_grid(images: List[torch.Tensor], out_path: str, nrow: int = 6):
    grid = make_grid(torch.cat(images, dim=0), nrow=nrow, normalize=True, value_range=(-1, 1))
    img = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    plt.imshow(img)
    plt.axis('off')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


def run_sweep(checkpoint: str,
              schedulers: List[str],
              cfg_scales: List[float],
              etas: List[float],
              labels: List[int],
              base_seed: int,
              out_dir: str):
    ckpt = torch.load(checkpoint, map_location=device)

    model = UNET(time_steps=500,
                 input_channels=3,
                 output_channels=3,
                 label_embedding=True,
                 use_film=False,
                 dropout_prob=0.1).to(device)
    model.load_state_dict(ckpt['weights'])
    ema = ModelEmaV2(model, decay=0.999)
    ema.load_state_dict(ckpt['ema'])
    model_eval = ema.module.eval()

    for sched_name in schedulers:
        if sched_name == 'linear':
            sched = DDPM_beta_t_linear_scheduler(500).to(device)
        else:
            sched = DDPM_beta_t_cosine_scheduler(500).to(device)

        for eta in etas:
            for cfg in cfg_scales:
                images = []
                for idx, lbl in enumerate(labels):
                    seed = base_seed + idx
                    g_scale = 0.0 if lbl == 10 else cfg
                    x0 = _sample_from_noise(model_eval, sched, lbl, g_scale, eta, num_steps=500, seed=seed)
                    images.append(x0)

                tag = f"{sched_name}_cfg{cfg}_eta{eta}"
                out_path = os.path.join(out_dir, sched_name, f"grid_{tag}.png")
                _save_grid(images, out_path, nrow=6)


def parse_args():
    p = argparse.ArgumentParser(description='Inference sweeps for scheduler/CFG/eta')
    p.add_argument('--checkpoint', type=str, default='stl10_checkpoint_epoch_3200.pt')
    p.add_argument('--schedulers', type=str, nargs='+', default=['cosine', 'linear'])
    p.add_argument('--cfg', type=float, nargs='+', default=[0.0, 1.5, 3.0, 6.0, 9.0])
    p.add_argument('--eta', type=float, nargs='+', default=[0.0, 1.0])
    p.add_argument('--labels', type=int, nargs='+', default=list(range(10)) + [10])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default=os.path.join('docs', 'images', 'sweeps'))
    return p.parse_args()


def main():
    args = parse_args()
    run_sweep(
        checkpoint=args.checkpoint,
        schedulers=args.schedulers,
        cfg_scales=args.cfg,
        etas=args.eta,
        labels=args.labels,
        base_seed=args.seed,
        out_dir=args.out
    )


if __name__ == '__main__':
    main()


