import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from config import LDM, IMAGE_NET
from VAE import get_vae

if IMAGE_NET:
    UNCOND_ID = 1000
else:
    UNCOND_ID = 10

STL10_CLASS_NAMES = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck','uncond']

IMAGENET_ID_TO_NAME = {
    19: 'airliner',
    363: 'goldfinch',
    829: 'sports car',
    872: 'tabby',
    353: 'gazelle',
    84: 'beagle',
    816: 'sorrel',
    531: 'macaque',
    233: 'container ship',
    915: 'trailer truck',
    1000: 'uncond'
}
def sample_noise(LDM = True, W = 96, H = 96, batch_size = 1, device = torch.device('cuda')) -> torch.Tensor:
    if IMAGE_NET:
        W = 256
        H = 256
    if LDM:
        return torch.randn(batch_size, 4, W//8, H//8, device=device)
    else:
        return torch.randn(batch_size, 3, W, H, device=device)

def tensor_to_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert an image tensor in [-1,1] to HxWxC numpy in [0,1].

    Accepts [N,C,H,W] or [C,H,W] or [H,W,C]. Returns HxWxC np.ndarray.
    """

    if LDM:
        if IMAGE_NET:
            W = 256
            H = 256
        if not img_tensor.ndim == 4:
            raise ValueError("Image tensor must be 4D")

        vae = get_vae()
        image_tensor = vae.decode(img_tensor)

    else:
        image_tensor = img_tensor
    img = image_tensor.detach().cpu()
    if img.ndim == 4:   
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        # [C,H,W] -> [H,W,C]
        img = img
    elif img.ndim == 2:
        img = img.unsqueeze(0)
    img = torch.clamp((img + 1) / 2, 0, 1)
    img = img.permute(1, 2, 0).numpy()
    
    return img


def save_denoising_collage(images: Sequence[torch.Tensor], timesteps: Sequence[int], out_path: str) -> None:
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 1.5, 1.5))
    if cols == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        img = images[idx]
        if isinstance(img, torch.Tensor):
            img = tensor_to_image(img)
        ax.imshow(img)
        ax.set_title(f"t={timesteps[idx]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


def _ddpm_step(z: torch.Tensor, t: int, predicted_noise: torch.Tensor, scheduler, eta: float) -> torch.Tensor:
    alpha_bar_t = scheduler.alpha_t[t]
    beta_t = scheduler.beta_t[t]
    alpha_t = 1.0 - beta_t
    if t > 0:
        alpha_bar_prev = scheduler.alpha_t[t - 1]
    else:
        alpha_bar_prev = torch.tensor(1.0, device=z.device)

    # DDPM/DDIM mean step
    z = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)

    # Add noise for t>1 using posterior variance scaled by eta (eta=0 -> deterministic DDIM)
    if t > 1 and eta > 0.0:
        noise = torch.randn_like(z)
        posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        z = z + eta * torch.sqrt(posterior_var) * noise
    return z


def run_reverse_process(
    model,
    scheduler,
    z_init: torch.Tensor,
    labels: torch.Tensor,
    guidance_scale: float,
    eta: float,
    num_time_steps: int,
    predict_eps_fn: Callable[[torch.nn.Module, object, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor],
    capture_timesteps: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Run reverse denoising from t_start to 0.

    Returns final x0 tensor and any requested snapshots in a dict by timestep.
    """
    with torch.no_grad():
        model.eval()
        if capture_timesteps is None:
            capture_timesteps = []
        snapshots: Dict[int, torch.Tensor] = {}

        # infer t_start from provided z shape vs scheduler length; caller should set z at the right noise level
        t_start = num_time_steps - 1
        # We allow callers to run partial schedules by preparing z_init accordingly and setting capture list
        z = z_init.clone()
        for t in reversed(range(1, t_start + 1)):
            t_tensor = torch.tensor([t], device=z.device)
            predicted_noise = predict_eps_fn(model, scheduler, z, t_tensor, labels, guidance_scale)
            z = _ddpm_step(z, t, predicted_noise, scheduler, eta)
            if t in capture_timesteps:
                snapshots[t] = torch.clamp(z, -1, 1)

        # final deterministic step t=0
        t_tensor = torch.tensor([0], device=z.device)
        predicted_noise = predict_eps_fn(model, scheduler, z, t_tensor, labels, guidance_scale)
        alpha_bar_0 = scheduler.alpha_t[0]
        beta_0 = scheduler.beta_t[0]
        alpha_0 = 1.0 - beta_0
        x0 = (1.0 / torch.sqrt(alpha_0)) * (z - (beta_0 / torch.sqrt(1.0 - alpha_bar_0)) * predicted_noise)
        snapshots[0] = torch.clamp(x0, -1, 1)
        return torch.clamp(x0, -1, 1), snapshots


def sample_from_noise(
    model,
    scheduler,
    label: int,
    guidance_scale: float,
    eta: float,
    num_time_steps: int,
    predict_eps_fn: Callable[[torch.nn.Module, object, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        z = sample_noise(LDM).to(device)
        labels = torch.tensor([label], device=device, dtype=torch.long)
        # Use the standard reverse process; we do not need snapshots here
        x0, _ = run_reverse_process(
            model,
            scheduler,
            z,
            labels,
            guidance_scale,
            eta,
            num_time_steps,
            predict_eps_fn,
            capture_timesteps=[],
        )
        return x0


def generate_denoise_collage(
    model,
    scheduler,
    epoch: int,
    step: int,
    timesteps: Optional[Sequence[int]],
    device: torch.device,
    num_time_steps: int,
    sampling_eta: float,
    predict_eps_fn: Callable[[torch.nn.Module, object, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor],
    output_dir: str,
    labels: Optional[torch.Tensor] = None,
    guidance_scale: float = 0.0,
) -> None:
    if timesteps is None:
        timesteps = [999, 700, 550, 400, 300, 200, 100, 50, 15, 0]
    with torch.no_grad():
        model.eval()
        z = sample_noise(LDM).to(device)    
        if labels is None:
            labels = torch.randint(0, 10, (1,), device=device)
        # Run and capture the requested timesteps
        _, snapshots = run_reverse_process(
            model,
            scheduler,
            z,
            labels,
            guidance_scale,
            sampling_eta,
            num_time_steps,
            predict_eps_fn,
            capture_timesteps=timesteps,
        )
        ordered_images = [snapshots[t] for t in timesteps if t in snapshots]
        if len(ordered_images) == 0:
            return
        out_dir = os.path.join(output_dir, f'epoch_{epoch}')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'epoch_{epoch}_step_{step}_stl10_collage.png')
        save_denoising_collage(ordered_images, [t for t in timesteps if t in snapshots], out_path)


def generate_random_sample_image(
    model,
    scheduler,
    epoch: int,
    device: torch.device,
    num_time_steps: int,
    sampling_eta: float,
    predict_eps_fn: Callable[[torch.nn.Module, object, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor],
    output_dir: str,
    labels: Optional[torch.Tensor] = None,
    guidance_scale: float = 0.0,
) -> None:
    with torch.no_grad():
        model.eval()
        z = sample_noise(LDM).to(device)
        if labels is None:
            labels = torch.randint(0, 10, (1,), device=device)
        x0, _ = run_reverse_process(
            model,
            scheduler,
            z,
            labels,
            guidance_scale,
            sampling_eta,
            num_time_steps,
            predict_eps_fn,
            capture_timesteps=[],
        )
        img_np = tensor_to_image(x0)
        plt.figure(figsize=(3, 3))
        plt.imshow(img_np)
        plt.title(f'STL10 sample - Epoch {epoch}')
        plt.axis('off')
        out_dir = os.path.join(output_dir, f'epoch_{epoch}')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'epoch_{epoch}_stl10_sample.png'), bbox_inches='tight', dpi=150)
        plt.close()


def reconstruct_from_dataset_sample(
    model,
    scheduler,
    dataset,
    epoch: int,
    device: torch.device,
    num_time_steps: int,
    sampling_eta: float,
    predict_eps_fn: Callable[[torch.nn.Module, object, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor],
    output_dir: str,
    t_start: int = 350,
    guidance_scale: float = 0.0,
) -> None:
    with torch.no_grad():
        model.eval()
        max_idx = len(dataset)
        idx = int(torch.randint(0, max_idx, (1,)).item())
        x, label = dataset[idx]
        x = x.unsqueeze(0).to(device)
        labels = torch.tensor([label], device=device, dtype=torch.long)
        if guidance_scale == 0.0:
            labels = torch.full_like(labels, UNCOND_ID)

        t0 = int(max(1, min(num_time_steps - 1, t_start)))
        a_bar = scheduler.alpha_t[t0].to(device).view(1, 1, 1, 1)
        e = torch.randn_like(x)
        x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * e

        # partial reverse from t0
        z = x_t.clone()
        # We reuse the same runner but with z already at t0; snapshots not needed here
        # For simplicity, walk manually from t0 to 0 mirroring run_reverse_process
        snapshots: Dict[int, torch.Tensor] = {}
        for t in reversed(range(1, t0 + 1)):
            t_tensor = torch.tensor([t], device=device)
            predicted_noise = predict_eps_fn(model, scheduler, z, t_tensor, labels, guidance_scale)
            z = _ddpm_step(z, t, predicted_noise, scheduler, sampling_eta)
        # final t=0
        t_tensor = torch.tensor([0], device=device)
        predicted_noise = predict_eps_fn(model, scheduler, z, t_tensor, labels, guidance_scale)
        alpha_bar_0 = scheduler.alpha_t[0]
        beta_0 = scheduler.beta_t[0]
        alpha_0 = 1.0 - beta_0
        x0_hat = (1.0 / torch.sqrt(alpha_0)) * (z - (beta_0 / torch.sqrt(1.0 - alpha_bar_0)) * predicted_noise)

        def to_img(tensor: torch.Tensor) -> np.ndarray:
            return tensor_to_image(torch.clamp(tensor, -1, 1))

        orig = to_img(x)
        xt_img = to_img(x_t)
        recon = to_img(x0_hat)

        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
        if IMAGE_NET:
            label_names = IMAGENET_ID_TO_NAME.values()
        else:
            label_names = STL10_CLASS_NAMES
        for ax, im, title in zip(axes, [orig, xt_img, recon], [label_names[label], f"x_t (t={t0})", "recon"]):
            ax.imshow(im)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        out_dir = os.path.join(output_dir, f'epoch_{epoch}')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'epoch_{epoch}_reconstruction_t{t0}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()


def reconstruct_sweep(
    model,
    scheduler,
    loader,
    epoch: int,
    device: torch.device,
    num_time_steps: int,
    sampling_eta: float,
    predict_eps_fn: Callable[[torch.nn.Module, object, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor],
    output_dir: str,
    t_list: Optional[Sequence[int]] = None,
    guidance_scale: float = 0.0,
) -> None:
    if t_list is None:
        t_list = [400, 300, 200, 100]
    with torch.no_grad():
        model.eval()
        x, label = next(iter(loader))
        x = x[0]
        label = label[0]
    
        x = x.unsqueeze(0).to(device)
        labels = torch.tensor([label], device=device, dtype=torch.long)
        if guidance_scale == 0.0:
            labels = torch.full_like(labels, UNCOND_ID)

        panels: List[np.ndarray] = []
        titles: List[str] = []
        panels.append(tensor_to_image(torch.clamp(x, -1, 1)))
        if IMAGE_NET:
            label_names = IMAGENET_ID_TO_NAME
        else:
            label_names = STL10_CLASS_NAMES
        titles.append(label_names.get(label.item()))

        for t0 in t_list:
            t0 = int(max(1, min(num_time_steps - 1, t0)))
            a_bar = scheduler.alpha_t[t0].to(device).view(1, 1, 1, 1)
            e = torch.randn_like(x)
            x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * e

            z = x_t.clone()
            for t in reversed(range(1, t0 + 1)):
                t_tensor = torch.tensor([t], device=device)
                predicted_noise = predict_eps_fn(model, scheduler, z, t_tensor, labels, guidance_scale)
                z = _ddpm_step(z, t, predicted_noise, scheduler, sampling_eta)
            t_tensor = torch.tensor([0], device=device)
            predicted_noise = predict_eps_fn(model, scheduler, z, t_tensor, labels, guidance_scale)
            alpha_bar_0 = scheduler.alpha_t[0]
            beta_0 = scheduler.beta_t[0]
            alpha_0 = 1.0 - beta_0
            x0_hat = (1.0 / torch.sqrt(alpha_0)) * (z - (beta_0 / torch.sqrt(1.0 - alpha_bar_0)) * predicted_noise)

            panels.append(tensor_to_image(torch.clamp(x0_hat, -1, 1)))
            titles.append(f'recon t={t0} {label_names.get(label.item())}')

        fig, axes = plt.subplots(1, len(panels), figsize=(2 * len(panels), 2))
        if len(panels) == 1:
            axes = [axes]
        for ax, im, title in zip(axes, panels, titles):
            ax.imshow(im)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        out_dir = os.path.join(output_dir, f'epoch_{epoch}')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'epoch_{epoch}_reconstruction_sweep.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()


def generate_fixed_seed_class_grid(
    model,
    scheduler,
    epoch: int,
    step: int,
    cfg_scale: float,
    eta: float,
    label_ids: List[int],
    label_names: dict,
    base_seed: int,
    device: torch.device,
    num_time_steps: int,
    predict_eps_fn: Callable[[torch.nn.Module, object, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor],
    output_dir: str,
) -> None:
    seeds = [base_seed + i for i in range(len(label_ids))]
    images: List[np.ndarray] = []
    titles: List[str] = []
    for idx, lbl in enumerate(label_ids):
        torch.manual_seed(seeds[idx])
        g_scale = 0.0 if lbl == UNCOND_ID else cfg_scale
        x0 = sample_from_noise(
            model,
            scheduler,
            lbl,
            g_scale,
            eta,
            num_time_steps,
            predict_eps_fn,
            device,
        )
        
        # Optional CLIP-based check: append predicted class to title if CLIP available
        title = label_names.get(lbl)
        try:
            pred_idx, score = utils.clip_predict_top1_class(torch.clamp(x0, -1, 1))
            title = f"{title} | pred: {label_names[pred_idx]} ({score:.2f})"
        except Exception:
            pass
        images.append(tensor_to_image(x0))
        titles.append(title)

    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].axis('off')
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
    plt.tight_layout()
    out_dir = os.path.join(output_dir, f'epoch_{epoch}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'epoch_{epoch}_step_{step}_fixed_seed_grid_cfg{cfg_scale}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


def save_loss_curve(step_history: Sequence[int], loss_history: Sequence[float], epoch: int, output_dir: str, suffix: str = "step") -> None:
    if len(step_history) == 0:
        return
    plt.figure(figsize=(5, 3))
    plt.plot(step_history, loss_history, linewidth=1.2)
    plt.xlabel('global step')
    plt.ylabel('train MSE loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    out_dir = os.path.join(output_dir, f'epoch_{epoch}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'loss_curve_{suffix}_{step_history[-1]}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


def sample_dpm_solver(
    model,
    scheduler,
    label_id: int,
    cfg_scale: float,
    num_inference_steps: int,
    device: torch.device,
    latent_shape: tuple = (1, 4, 32, 32),  # 256÷8=32 for your setup
) -> torch.Tensor:
    """Sample from noise using DPM-Solver++ with CFG for v-prediction.
    
    Returns decoded image tensor in [-1, 1] range, ready for tensor_to_image().
    """
    # Initialize random latent
    latents = torch.randn(latent_shape, device=device)
    
    # Get context embeddings
    labels = torch.tensor([label_id], device=device)
    context = utils._labels_to_context(labels)
    null_labels = torch.full_like(labels, UNCOND_ID)
    null_context = utils._labels_to_context(null_labels)
    
    # Set timesteps for DPM-Solver++
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    model.eval()
    with torch.no_grad():
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Prepare timestep tensor
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            
            if cfg_scale > 1.0:
                # Expand latents for conditional + unconditional
                latent_model_input = torch.cat([latents, latents])
                t_input = torch.cat([t_tensor, t_tensor])
                
                # Get v-predictions
                v_uncond = model(latent_model_input[:1], t_input[:1], null_context)
                v_cond = model(latent_model_input[1:], t_input[1:], context)
                
                # Apply CFG
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                # No guidance
                v_pred = model(latents, t_tensor, context)
            
            # DPM-Solver++ expects velocity prediction, which is what we have
            # The scheduler handles the conversion internally
            latents = scheduler.step(v_pred, t, latents).prev_sample
    
    # Decode latents using VAE (your tensor_to_image expects latents if LDM=True)
    # Return latents, not decoded images, since tensor_to_image does the decoding
    return latents


def generate_fixed_seed_class_grid_dpm(
    model,
    scheduler,
    epoch: int,
    step: int,
    cfg_scale: float,
    num_inference_steps: int,
    label_ids: List[int],
    label_names: dict,
    base_seed: int,
    device: torch.device,
    output_dir: str,
) -> None:
    """Generate class grid using DPM-Solver++ sampling.
    
    Compatible with your existing tensor_to_image() function.
    """
    model.eval()
    seeds = [base_seed + i for i in range(len(label_ids))]
    images: List[np.ndarray] = []
    titles: List[str] = []
    
    for idx, lbl in enumerate(label_ids):
        torch.manual_seed(seeds[idx])
        
        # Skip CFG for unconditional
        g_scale = 0.0 if lbl == UNCOND_ID else cfg_scale
        
        # Sample latents using DPM-Solver++
        latents = sample_dpm_solver(
            model,
            scheduler,
            lbl,
            g_scale,
            num_inference_steps,
            device,
        )
        
        # Convert to image using your existing function
        # tensor_to_image will handle VAE decoding since LDM=True
        img_np = tensor_to_image(latents)
        
        # Optional CLIP prediction
        title = label_names.get(lbl, str(lbl))
        try:
            # Reconstruct tensor in [-1,1] for CLIP
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor * 2 - 1  # [0,1] -> [-1,1]
            pred_idx, score = utils.clip_predict_top1_class(img_tensor.to(device))
            title = f"{title} | pred: {label_names.get(pred_idx, str(pred_idx))} ({score:.2f})"
        except Exception:
            pass
        
        images.append(img_np)
        titles.append(title)

    # Plot grid
    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = axes.flatten()
    
    for i in range(rows * cols):
        axes[i].axis('off')
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    out_dir = os.path.join(output_dir, f'epoch_{epoch}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, 
        f'epoch_{epoch}_step_{step}_dpm_solver_cfg{cfg_scale}_steps{num_inference_steps}.png'
    )
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out_path}")