import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import re
from unet import UNET
from tqdm import tqdm
from timm.utils import ModelEmaV2
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from typing import Optional
from visualization import (
    tensor_to_image as viz_tensor_to_image,
    save_denoising_collage as viz_save_denoising_collage,
    generate_denoise_collage as viz_generate_denoise_collage,
    generate_random_sample_image as viz_generate_random_sample_image,
    reconstruct_from_dataset_sample as viz_reconstruct_from_dataset_sample,
    reconstruct_sweep as viz_reconstruct_sweep,
    generate_fixed_seed_class_grid as viz_generate_fixed_seed_class_grid,
    save_loss_curve as viz_save_loss_curve,
    sample_noise as viz_sample_noise,
)
from utils import *
import utils

# Hyperparameters
NUM_EPOCHS = 4000
BATCH_SIZE = 64
LR = 1e-4
EMA_DECAY = 0.999
NUM_TIME_STEPS = 500
DROPOUT = 0.1
from config import LDM

# Extended training (resume 2000 -> 4000)
TOTAL_EPOCHS_NEW = 4000
START_EPOCH = 0

# Debug/overfit tiny subset
OVERFIT_TINY = False
OVERFIT_SIZE = 512  # number of images to overfit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# Directory for saving intermediate images
OUTPUT_DIR = 'intermiate imagees'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classifier-Free Guidance hyperparameters
CFG_SCALE = 6.0           # guidance strength at sampling
CFG_P_UNCOND = 0.2        # probability to drop label to null during training
SAMPLING_ETA = 1.0        # 1.0 ~ DDPM (stochastic), 0.0 ~ DDIM (deterministic)

# Monitoring configuration
FIXED_SEED = 42
FIXED_SEED_INTERVAL_STEPS = 100
CFG_MONITOR_SCALE = 1.5
STL10_CLASS_NAMES = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck','uncond']
LOSS_PLOT_INTERVAL_STEPS = 100
VIRTUAL_BATCH_SIZE_MULTIPLIER = 2  # set >1 to accumulate this many micro-batches per optimizer step

def _tensor_to_image(img_tensor: torch.Tensor) -> np.ndarray:
    return viz_tensor_to_image(img_tensor)


def save_denoising_collage(images, timesteps, filename):
    viz_save_denoising_collage(images, timesteps, filename)


def _cfg_predict_eps_from_v(model, scheduler, z, t_tensor, labels, guidance_scale: float):
    """Classifier-free guidance on v-prediction, then convert to eps.

    v = sqrt(a_bar)*eps - sqrt(1-a_bar)*x0
    eps = sqrt(1-a_bar)*x_t + sqrt(a_bar)*v
    (since a_bar + (1-a_bar) = 1)
    """
    # Unconditional and conditional v predictions
    null_labels = torch.full_like(labels, 10)
    null_context = utils._labels_to_context(null_labels)
    v_uncond = model(z, t_tensor, null_context)
    context = utils._labels_to_context(labels)
    v_cond = model(z, t_tensor, context)
    v_hat = v_uncond + guidance_scale * (v_cond - v_uncond)

    # Convert v -> eps using current timestep coefficients
    alpha_bar_t = scheduler.alpha_t[t_tensor]
    alpha_t_sqrt = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
    sigma_t = torch.sqrt(1.0 - alpha_bar_t).view(-1, 1, 1, 1)
    eps_hat = sigma_t * z + alpha_t_sqrt * v_hat
    return eps_hat


def generate_denoise_collage(model, scheduler, epoch, step, timesteps=None, labels=None, guidance_scale: float = CFG_SCALE):
    viz_generate_denoise_collage(
        model,
        scheduler,
        epoch,
        step,
        timesteps,
        device,
        NUM_TIME_STEPS,
        SAMPLING_ETA,
        _cfg_predict_eps_from_v,
        OUTPUT_DIR,
        labels=labels,
        guidance_scale=guidance_scale,
    )


def generate_random_sample_image(model, scheduler, epoch, labels=None, guidance_scale: float = CFG_SCALE):
    viz_generate_random_sample_image(
        model,
        scheduler,
        epoch,
        device,
        NUM_TIME_STEPS,
        SAMPLING_ETA,
        _cfg_predict_eps_from_v,
        OUTPUT_DIR,
        labels=labels,
        guidance_scale=guidance_scale,
    )


def reconstruct_from_dataset_sample(model, scheduler, dataset, epoch, t_start: int = 350, guidance_scale: float = 0.0):
    viz_reconstruct_from_dataset_sample(
        model,
        scheduler,
        dataset,
        epoch,
        device,
        NUM_TIME_STEPS,
        SAMPLING_ETA,
        _cfg_predict_eps_from_v,
        OUTPUT_DIR,
        t_start=t_start,
        guidance_scale=guidance_scale,
    )


def reconstruct_sweep(model, scheduler, dataset, epoch, t_list=None, guidance_scale: float = 0.0):
    viz_reconstruct_sweep(
        model,
        scheduler,
        dataset,
        epoch,
        device,
        NUM_TIME_STEPS,
        SAMPLING_ETA,
        _cfg_predict_eps_from_v,
        OUTPUT_DIR,
        t_list=t_list,
        guidance_scale=guidance_scale,
    )

def _sample_from_noise(model, scheduler, label: int, guidance_scale: float, eta: float):
    from visualization import sample_from_noise as viz_sample_from_noise
    return viz_sample_from_noise(
        model,
        scheduler,
        label,
        guidance_scale,
        eta,
        NUM_TIME_STEPS,
        _cfg_predict_eps_from_v,
        device,
    )

def generate_fixed_seed_class_grid(model, scheduler, epoch, step, cfg_scale: float = CFG_MONITOR_SCALE, eta: float = 0.0,
                                   label_ids: list = None, label_names: list = None):
    if label_ids is None:
        label_ids = list(range(10)) + [10]
    if label_names is None:
        label_names = STL10_CLASS_NAMES[:len(label_ids)] if len(STL10_CLASS_NAMES) >= len(label_ids) else [str(i) for i in label_ids]
    viz_generate_fixed_seed_class_grid(
        model,
        scheduler,
        epoch,
        step,
        cfg_scale,
        eta,
        label_ids,
        label_names,
        FIXED_SEED,
        device,
        NUM_TIME_STEPS,
        _cfg_predict_eps_from_v,
        OUTPUT_DIR,
    )

def save_loss_curve(step_history, loss_history, epoch, suffix: str = "step"):
    viz_save_loss_curve(step_history, loss_history, epoch, OUTPUT_DIR, suffix=suffix)

def _build_loader_overfit_subset(dataset, batch_size):
    indices = list(range(min(OVERFIT_SIZE, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)


def train(checkpoint_path=None, clip_table_path: str = os.path.join('artifacts', 'clip', 'clip_text_emb_stl10_vitb32.pt')):
    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)

    train_data = load_dataset(LDM = LDM)

    if OVERFIT_TINY:
        train_loader = _build_loader_overfit_subset(train_data, BATCH_SIZE)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Load CLIP context table once BEFORE building model so context_dim is set
    global CONTEXT_TABLE
    if CONTEXT_TABLE is None and os.path.exists(clip_table_path):
        CONTEXT_TABLE = utils._load_clip_context_table(clip_table_path)
        print(f"[info] Loaded CLIP context table from {clip_table_path} with dim {CONTEXT_TABLE.shape[-1]}")
    elif CONTEXT_TABLE is None:
        print(f"[warn] CLIP table not found at {clip_table_path}; falling back to label ids as context")

    model = load_model(LDM = LDM, NUM_TIME_STEPS = 1000, DROPOUT = 0.1)
    # Sanity: confirm cross-attention is enabled when CLIP table is present
    try:
        if utils.CONTEXT_TABLE is not None:
            table_shape = tuple(utils.CONTEXT_TABLE.shape)
            expected_labels = 11
            if table_shape[0] != expected_labels:
                print(f"[warn] CLIP context table first dim is {table_shape[0]}, expected {expected_labels} (10 classes + uncond)")
            # Count cross-attention modules attached to layers
            num_ca = 0
            if hasattr(model, 'num_layers'):
                for li in range(model.num_layers):
                    layer = getattr(model, f'Layer{li+1}', None)
                    if layer is not None and hasattr(layer, 'cross_attention'):
                        num_ca += 1
            print(f"[info] Cross-attention modules active: {num_ca}; CLIP table shape: {table_shape}")
        else:
            print("[warn] CONTEXT_TABLE is None; sampling will ignore text conditioning.")
    except Exception as e:
        print(f"[warn] Cross-attn/CLIP sanity check skipped: {e}")
    model = model.to(device)

    ema = ModelEmaV2(model, decay=EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.99))
    criterion = nn.MSELoss(reduction='mean')

    collage_timesteps = [499, 400, 300, 200, 100, 50, 15, 0]

    # New: Warmup + Cosine schedule for extended run (2000 -> 4000 epochs)
    total_steps_new = TOTAL_EPOCHS_NEW * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps_new))  # 5% warmup on new total

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps_new - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    optimizer.zero_grad(set_to_none=True)

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            try:
                model.load_state_dict(checkpoint['weights'])
            except Exception as e:
                print(f"[warn] Exact state_dict load failed ({e}); retrying with strict=False")
                model.load_state_dict(checkpoint['weights'], strict=False)
            try:
                ema.load_state_dict(checkpoint['ema'])
            except Exception as e:
                print(f"[warn] EMA state_dict load failed ({e}); initializing EMA from model.")
                with torch.no_grad():
                    ema._clone_model_state()
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Checkpoint file not found: {checkpoint_path}")

    # Fast-forward LR scheduler to reflect completed epochs
    # Scheduler is stepped after each optimizer step (with grad accumulation).
    effective_steps_per_epoch = int((len(train_loader) + max(1, VIRTUAL_BATCH_SIZE_MULTIPLIER) - 1) // max(1, VIRTUAL_BATCH_SIZE_MULTIPLIER))
    steps_so_far = START_EPOCH * effective_steps_per_epoch
    for _ in range(steps_so_far):
        lr_scheduler.step()

    global_step = steps_so_far
    for epoch in range(START_EPOCH, TOTAL_EPOCHS_NEW):
        sum_loss = 0
        for i, (x, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS_NEW}")):
            x = x.to(device)
            labels = labels.to(device)
            # Classifier-free guidance training: randomly drop label to null class 10
            drop_mask = (torch.rand(labels.shape[0], device=device) < CFG_P_UNCOND)
            labels_dropped = labels.clone()
            labels_dropped[drop_mask] = 10
            t = torch.randint(0, NUM_TIME_STEPS, (x.shape[0],), requires_grad=False)
            a_bar = scheduler.alpha_t[t].to(device).view(-1, 1, 1, 1)
            e = torch.randn_like(x, requires_grad=False).to(device)
            x_t = (torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * e)

            # v-prediction target and loss
            alpha_bar = scheduler.alpha_t[t].to(device).view(-1, 1, 1, 1)
            sigma_t = torch.sqrt(1.0 - alpha_bar)
            alpha_t_sqrt = torch.sqrt(alpha_bar)
            v_target = alpha_t_sqrt * e - sigma_t * x

            # Build context vectors: CLIP embeddings for labels (10=uncond)
            if CONTEXT_TABLE is not None:
                context = utils._labels_to_context(labels_dropped)  # [B, D]
            else:
                # fallback: integer labels (old path)
                context = labels_dropped

            output = model(x_t, t, context)  # predict v
            loss = criterion(output, v_target)
            if not torch.isfinite(loss):
                print("[warn] Non-finite loss, skipping step")
                continue

            # Gradient accumulation
            scaled_loss = loss / max(1, VIRTUAL_BATCH_SIZE_MULTIPLIER)
            scaled_loss.backward()
            sum_loss += loss.item()

            do_step = ((i + 1) % max(1, VIRTUAL_BATCH_SIZE_MULTIPLIER) == 0) or (i + 1 == len(train_loader))
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update(model)
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # TensorBoard scalars at real optimizer step
                # writer.add_scalar("train/loss", loss.item(), global_step)
                # writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
                global_step += 1



        print(f'Epoch {epoch+1} | Loss {sum_loss / len(train_loader):.5f}')
        if (epoch + 1) % LOSS_PLOT_INTERVAL_STEPS == 0:
            generate_denoise_collage(ema.module, scheduler, epoch+1, 1, collage_timesteps)
            generate_fixed_seed_class_grid(ema.module, scheduler, epoch+1, global_step, cfg_scale=CFG_SCALE, eta=SAMPLING_ETA)
            reconstruct_from_dataset_sample(ema.module, scheduler, train_data, epoch+1, t_start=700, guidance_scale=CFG_SCALE)
            reconstruct_sweep(ema.module, scheduler, train_data, epoch+1, t_list=[700,500,300,200], guidance_scale=CFG_SCALE)

        out_mean = output.mean().item()
        out_std = output.std().item()
        e_mean = e.mean().item()
        e_std = e.std().item()
        print(f"[dbg] step {i}: loss={loss.item():.5f} | out(mu,sd)=({out_mean:.3f},{out_std:.3f}) | e(mu,sd)=({e_mean:.3f},{e_std:.3f})")

        if (epoch + 1) % 200 == 0:
            checkpoint_epoch = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }
            torch.save(checkpoint_epoch, f'stl10_checkpoint_epoch_{epoch+1}.pt')
        
        if (epoch + 1) % 500 == 0:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }
            torch.save(checkpoint, f'stl10_checkpoint_epoch_{epoch+1}.pt')

    
    # writer.flush()
    # writer.close()



def inference(checkpoint_path, label: int = None, guidance_scale: float = CFG_SCALE,
              clip_table_path: str = os.path.join('artifacts', 'clip', 'clip_text_emb_stl10_vitb32.pt')):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Load CLIP context table (for CFG sampling conditions later if needed)
    global CONTEXT_TABLE
    if CONTEXT_TABLE is None and os.path.exists(clip_table_path):
        CONTEXT_TABLE = utils._load_clip_context_table(clip_table_path)

    model = load_model(LDM = LDM, NUM_TIME_STEPS = 1000, DROPOUT = 0.1)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV2(model, decay=EMA_DECAY)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)

    with torch.no_grad():
        generate_denoise_collage(ema.module, scheduler, 1, 1, collage_timesteps)
        generate_fixed_seed_class_grid(ema.module, scheduler, 1, global_step, cfg_scale=CFG_SCALE, eta=SAMPLING_ETA)
        reconstruct_from_dataset_sample(ema.module, scheduler, train_data, 1, t_start=700, guidance_scale=CFG_SCALE)
        reconstruct_sweep(ema.module, scheduler, train_data, 1, t_list=[700,500,300,200], guidance_scale=CFG_SCALE)

def _parse_epoch_from_checkpoint_path(path: str) -> int:
    """Best-effort extract epoch number from checkpoint filename.
    Returns 0 if none is found.
    """
    try:
        base = os.path.basename(path)
        m = re.search(r"epoch_(\d+)", base)
        if m:
            return int(m.group(1))
        m2 = re.search(r"(\d+)", base)
        if m2:
            return int(m2.group(1))
    except Exception:
        pass
    return 0


def inference_full(checkpoint_path: str):
    """Load checkpoint and generate the same monitoring outputs as training.

    Outputs under `intermiate imagees/epoch_{epoch}/`:
    - Denoising collage
    - Fixed-seed class grid (classes 0-9 + unconditional)
    - Reconstruction from dataset sample (t=700)
    - Reconstruction sweep (t in {700, 500, 300, 200})
    - t-SNE of label embeddings
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Ensure CLIP context is loaded BEFORE building model so cross-attn context_dim is correct
    clip_table_path = os.path.join('artifacts', 'clip', 'clip_text_emb_stl10_vitb32.pt')
    global CONTEXT_TABLE
    if CONTEXT_TABLE is None and os.path.exists(clip_table_path):
        CONTEXT_TABLE = utils._load_clip_context_table(clip_table_path)

    model = load_model(LDM = LDM, NUM_TIME_STEPS = 1000, DROPOUT = 0.1)
    # Load model and EMA state if present
    if 'weights' in checkpoint:
        model.load_state_dict(checkpoint['weights'])
    else:
        model.load_state_dict(checkpoint)

    ema = ModelEmaV2(model, decay=EMA_DECAY)
    if isinstance(checkpoint, dict) and 'ema' in checkpoint:
        ema.load_state_dict(checkpoint['ema'])
    else:
        # Fallback: initialize EMA from current model weights
        with torch.no_grad():
            ema._clone_model_state()

    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)

    # Dataset for reconstruction utilities
    base_data = load_dataset(LDM = LDM)

    # Match training choices
    collage_timesteps = [499, 400, 300, 200, 100, 50, 15, 0]
    t_list = [700, 500, 300, 200]
    epoch = _parse_epoch_from_checkpoint_path(checkpoint_path)
    step = 1

    # Generate outputs using EMA weights
    generate_denoise_collage(ema.module, scheduler, epoch, step, collage_timesteps)
    generate_fixed_seed_class_grid(ema.module, scheduler, epoch, step, cfg_scale=CFG_SCALE, eta=SAMPLING_ETA)
    reconstruct_from_dataset_sample(ema.module, scheduler, base_data, epoch, t_start=700, guidance_scale=CFG_SCALE)
    reconstruct_sweep(ema.module, scheduler, base_data, epoch, t_list=t_list, guidance_scale=CFG_SCALE)

def main():
    train()
    # inference_full('stl10_checkpoint_epoch_800.pt')


if __name__ == '__main__':
    main()


