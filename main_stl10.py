import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from collections import Counter, defaultdict
import re
from unet import UNET
from tqdm import tqdm
from timm.utils import ModelEmaV2
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from typing import Optional, List
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
    generate_fixed_seed_class_grid_dpm as viz_generate_fixed_seed_class_grid_dpm,
)
from utils import *
import utils
from VAE import get_vae
from vae_encoder import denormalize

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LR = 5e-5
EMA_DECAY = 0.995
NUM_TIME_STEPS = 500
DROPOUT = 0.1
from config import LDM, IMAGE_NET, USING_LPIPS

if IMAGE_NET:
    CLIP_TABLE_PATH = 'artifacts/clip/clip_text_emb_imagenet_vitb32.pt'
else:
    CLIP_TABLE_PATH = 'artifacts/clip/clip_text_emb_stl10_vitb32.pt'

# Extended training (resume 2000 -> 4000)
TOTAL_EPOCHS_NEW = 100
START_EPOCH = 20

# Debug/overfit tiny subset
OVERFIT_TINY = False
OVERFIT_SIZE = 512  # number of images to overfit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if IMAGE_NET:
    UNCOND_LABEL = 'uncond'
    UNCOND_ID = 1000

print(device)

# Directory for saving intermediate images
OUTPUT_DIR = 'intermiate imagees'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classifier-Free Guidance hyperparameters
CFG_SCALE = 8.0           # guidance strength at sampling
CFG_P_UNCOND = 0.2        # probability to drop label to null during training
SAMPLING_ETA = 1.0        # 1.0 ~ DDPM (stochastic), 0.0 ~ DDIM (deterministic)

# Monitoring configuration
FIXED_SEED = 42
FIXED_SEED_INTERVAL_STEPS = 100
CFG_MONITOR_SCALE = 1.5
STL10_CLASS_NAMES = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck','uncond']
IMAGE_NET_CLASS_NAMES  = [
    'airliner',        # For 'airplane'
    'goldfinch',       # For 'bird'
    'sports_car',      # For 'car'
    'tabby_cat',       # For 'cat'
    'gazelle',         # For 'deer'
    'beagle',          # For 'dog'
    'sorrel',          # For 'horse'
    'macaque',         # For 'monkey'
    'container_ship',  # For 'ship'
    'pickup_truck',     # For 'truck'
    'uncond'
]

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

LOSS_PLOT_INTERVAL_STEPS = 1
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
    null_labels = torch.full_like(labels, UNCOND_ID)
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
                                   label_ids: list = None, label_names: dict = None):
    if IMAGE_NET:
        label_ids = list(IMAGENET_ID_TO_NAME.keys())  
    else:
        label_ids = list(range(10)) + [10]
    if label_names is None:
        if IMAGE_NET:
            label_names = IMAGENET_ID_TO_NAME
        else:
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
    """
    Creates a DataLoader for a small, potentially class-balanced subset of the dataset.
    This is useful for quick overfitting tests to ensure the model can learn.
    """



    if not IMAGE_NET:
        # Default behavior for non-ImageNet: just take the first N samples
        print(f"[info] Creating a simple overfit subset of the first {OVERFIT_SIZE} samples.")
        indices = list(range(min(OVERFIT_SIZE, len(dataset))))
        subset = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

    # --- Class-Balanced Subset Logic for ImageNet ---
    print(f"[info] Creating a balanced ImageNet subset of ~{OVERFIT_SIZE} samples from 10 classes.")

    # --- Step 1: Get all labels from the dataset in a consistent list format ---
    all_labels = []
    try:
        if hasattr(dataset, 'targets'):  # For datasets like ImageNet, CIFAR10
            all_labels = dataset.targets
        elif hasattr(dataset, 'labels'): # For datasets from Hugging Face or custom ones
            all_labels = dataset.labels
        else:
            raise AttributeError("Dataset has no '.targets' or '.labels' attribute for class-based subsetting.")

        if torch.is_tensor(all_labels):
            all_labels = all_labels.tolist()
        all_labels = list(map(int, all_labels)) # Ensure all labels are integers

    except Exception as e:
        print(f"[Warning] Could not retrieve labels to create a balanced subset: {e}")
        print("[info] Falling back to a simple random subset.")
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        indices = indices[:min(OVERFIT_SIZE, len(indices))]
        subset = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

    # --- Step 2: Group indices by their class ID for the first 10 classes ---
    ALLOWED_CLASSES = set(IMAGENET_ID_TO_NAME.keys())
    NUM_CLASSES = len(ALLOWED_CLASSES)
    indices_by_class = defaultdict(list)
    for i, label in enumerate(all_labels):
        if label in ALLOWED_CLASSES:
            indices_by_class[label].append(i)

    if not indices_by_class:
        print("[Error] No samples found for the desired classes (0-9). Cannot create subset.")
        # Fallback to a simple slice to avoid crashing
        indices = list(range(min(OVERFIT_SIZE, len(dataset))))
        subset = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

    # --- Step 3: Sample indices from each class to create a balanced set ---
    final_indices = []
    samples_per_class = OVERFIT_SIZE // NUM_CLASSES
    remainder = OVERFIT_SIZE % NUM_CLASSES

    for class_id in sorted(indices_by_class.keys()):
        class_indices = indices_by_class[class_id]
        np.random.shuffle(class_indices) # Shuffle to get random samples from this class

        # Distribute the remainder evenly among the first few classes
        num_to_take = samples_per_class
        if remainder > 0:
            num_to_take += 1
            remainder -= 1

        # Take samples, but not more than are available for that class
        num_to_take = min(num_to_take, len(class_indices))
        final_indices.extend(class_indices[:num_to_take])

    # --- Step 4: Final shuffle and subset creation ---
    # This is crucial so that batches during training are not ordered by class
    np.random.shuffle(final_indices)
    subset = torch.utils.data.Subset(dataset, final_indices)

    # --- Step 5: Print the final class distribution in the created subset ---
    print("-" * 40)
    print(f"[info] Created a balanced subset with {len(subset)} images.")
    
    try:
        # Get labels corresponding to the final indices to count them
        subset_labels = [all_labels[i] for i in final_indices]
        counts = Counter(subset_labels)
        
        # Get human-readable names if available
        desired_names = IMAGENET_ID_TO_NAME

        print("[info] Final subset class counts:")
        for class_id, count in sorted(counts.items()):
            # Map class_id to a name, with a fallback
            name = desired_names[class_id] 
            print(f"  - {name}: {count}")
    except Exception as e:
        print(f"[Warning] Could not print class counts: {e}")
    
    print("-" * 40)

    # --- Step 6: Return the DataLoader ---
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)


def train(checkpoint_path=None, clip_table_path: str = CLIP_TABLE_PATH):
    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)

    train_data = load_dataset(LDM = LDM, IMAGE_NET = IMAGE_NET)

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
            if IMAGE_NET:
                expected_labels = 1001
            else:
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
    if config.USING_LOW_T_REWARDING:
        criterion = nn.MSELoss(reduction='none')
    else:
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
    if USING_LPIPS:
        vae = get_vae()
        lpips_model = utils.load_lpips_model()
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
            labels_dropped[drop_mask] = UNCOND_ID
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
            

            if config.USING_LOW_T_REWARDING and epoch >= config.LOW_T_REWARDING_WARMUP_EPOCHS:
                loss = criterion(output, v_target).mean(dim=(1,2,3))
                loss = loss * torch.exp(- config.LOW_T_REWARDING_WARMUP_WEIGHT * t.float()).to(device)
                loss = loss.mean()
            else:
                loss = criterion(output, v_target).mean()

            if not torch.isfinite(loss):
                print("[warn] Non-finite loss, skipping step")
                continue

            # if USING_LPIPS:
            #     x0_pred = torch.sqrt(a_bar) * x_t - torch.sqrt(1 - a_bar) * output
            #     x0_true = torch.sqrt(a_bar) * x_t - torch.sqrt(1 - a_bar) * v_target

            #     with torch.no_grad():
            #         img_pred = vae.decode(x0_pred )  # or .mode() if using diffusers
            #         img_true = vae.decode(x0_true )

            #     lpips_loss = lpips_model(img_pred, img_true).mean()
            #     loss = loss + 0.05 *lpips_loss

            

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
            reconstruct_sweep(ema.module, scheduler, train_loader, epoch+1, t_list=[700,500,300,200], guidance_scale=CFG_SCALE)
            generate_denoise_collage(ema.module, scheduler, epoch+1, 1, collage_timesteps)
            generate_fixed_seed_class_grid(ema.module, scheduler, epoch+1, global_step, cfg_scale=CFG_SCALE, eta=SAMPLING_ETA)
            # reconstruct_from_dataset_sample(ema.module, scheduler, train_data, epoch+1, t_start=700, guidance_scale=CFG_SCALE)

        out_mean = output.mean().item()
        out_std = output.std().item()
        e_mean = e.mean().item()
        e_std = e.std().item()
        print(f"[dbg] step {i}: loss={loss.item():.5f} | out(mu,sd)=({out_mean:.3f},{out_std:.3f}) | e(mu,sd)=({e_mean:.3f},{e_std:.3f})")

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict()
            }
            torch.save(checkpoint, f'stl10_checkpoint_epoch_{epoch+1}.pt')

    
    # writer.flush()
    # writer.close()



def inference(checkpoint_path, label: int = None, guidance_scale: float = CFG_SCALE,
              clip_table_path: str = CLIP_TABLE_PATH):
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
        # reconstruct_from_dataset_sample(ema.module, scheduler, train_data, 1, t_start=700, guidance_scale=CFG_SCALE)
        reconstruct_sweep(ema.module, scheduler, train_loader, 1, t_list=[700,500,300,200], guidance_scale=CFG_SCALE)

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


def inference_full(checkpoint_path: str, clip_table_path: str = CLIP_TABLE_PATH):
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
    clip_table_path = CLIP_TABLE_PATH
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
    # reconstruct_from_dataset_sample(ema.module, scheduler, base_data, epoch, t_start=700, guidance_scale=CFG_SCALE)
    reconstruct_sweep(ema.module, scheduler, train_loader, epoch, t_list=t_list, guidance_scale=CFG_SCALE)

def generate_cfg_grids_dpm_solver(
    checkpoint_path: str,
    cfg_scales: Optional[List[float]] = None,
    num_inference_steps: int = 50,
    clip_table_path: str = CLIP_TABLE_PATH,
):
    """Load a checkpoint and generate fixed-seed class grids using DPM-Solver++ sampling.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        cfg_scales: List of CFG scales to render. Defaults to [3.0, 5.0, 7.0].
        num_inference_steps: Number of DPM-Solver++ steps (20-50 recommended).
        clip_table_path: Path to CLIP text embedding table.
    """
    if cfg_scales is None:
        cfg_scales = [3.0, 5.0, 7.0]

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Ensure CLIP context is loaded
    global CONTEXT_TABLE
    if CONTEXT_TABLE is None and os.path.exists(clip_table_path):
        CONTEXT_TABLE = utils._load_clip_context_table(clip_table_path)

    model = load_model(LDM=LDM, NUM_TIME_STEPS=NUM_TIME_STEPS, DROPOUT=DROPOUT)

    # Load model and EMA state
    if isinstance(checkpoint, dict) and 'weights' in checkpoint:
        model.load_state_dict(checkpoint['weights'])
    else:
        model.load_state_dict(checkpoint)

    ema = ModelEmaV2(model, decay=EMA_DECAY)
    if isinstance(checkpoint, dict) and 'ema' in checkpoint:
        ema.load_state_dict(checkpoint['ema'])
    else:
        with torch.no_grad():
            ema._clone_model_state()

    # Create DPM-Solver++ scheduler
    from diffusers import DPMSolverMultistepScheduler
    
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=NUM_TIME_STEPS,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",  # Match your training
        algorithm_type="dpmsolver++",
        solver_order=2,
    )

    epoch = _parse_epoch_from_checkpoint_path(checkpoint_path)
    step = 1

    # Generate a grid per CFG scale using DPM-Solver++
    for cfg in cfg_scales:
        viz_generate_fixed_seed_class_grid_dpm(
            ema.module, 
            scheduler, 
            epoch, 
            step, 
            cfg_scale=cfg, 
            num_inference_steps=num_inference_steps,
            label_ids = list(IMAGENET_ID_TO_NAME.keys()),
            label_names = IMAGENET_ID_TO_NAME,
            base_seed = 42,
            device = device,
            output_dir = "intermiate imagees/dpm"
        )

def main():
    train(checkpoint_path='stl10_checkpoint_epoch_20.pt')
    # generate_cfg_grids_dpm_solver(
    # checkpoint_path='stl10_checkpoint_epoch_55.pt',
    # cfg_scales=[3.0, 4.0, 5.0, 6.0, 7.0],
    # num_inference_steps=100,  # Much faster than DDPM 500!
    # )


if __name__ == '__main__':
    main()


