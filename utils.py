import torchvision
import torchvision.transforms as transforms
import torch
import os
from unet import UNET
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed as dist_data
import torch.utils.data.sampler as sampler
import torch.utils.data.dataloader as dataloader
from typing import Optional
from config import LDM
from VAE import get_vae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CLIP context table (tested in main_stl10.py)
CONTEXT_TABLE: Optional[torch.Tensor] = None
CLIP_TABLE_PATH = 'docs/data/clip_table.pt'
CONTEXT_TABLE_DIM = 512
CONTEXT_TABLE_LABELS = 11

if LDM:
    # This will create the VAE instance the first time it's called,
    # and return the existing instance on subsequent calls.
    vae = get_vae()
    print("VAE is loaded and ready.")
else:
    vae = None
    print("Running without VAE (image-space model).")

class LatentDataset(data.Dataset):
    """
    A PyTorch Dataset for loading pre-computed VAE latents and labels
    from a .pt file.
    """
    def __init__(self, latent_path: str):
        print(f"Loading pre-computed latents from {latent_path}...")
        data_dict = torch.load(latent_path, map_location='cpu')
        self.scaling_factor = vae.scaling_factor
        self.latents = data_dict['latents']
        self.labels = data_dict['labels']
        
        print(f"Loaded {len(self.latents)} samples.")
        print(f"Latents shape: {self.latents.shape}")
        print(f"Labels shape: {self.labels.shape}")

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        scaled_latent = self.latents[idx] * self.scaling_factor
        return scaled_latent, self.labels[idx]


def load_dataset(LDM = False):
    if LDM:
        return LatentDataset(latent_path=os.path.join('docs', 'data', 'stl10_latents.pt'))
    else:
        TRANS = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.STL10(root='stl10_data', split='train', download=True, transform=TRANS)
        return dataset

def load_model(LDM = False, NUM_TIME_STEPS = 1000, DROPOUT = 0.1, use_flash_attention: bool = True):
    # Enable cross-attention conditioning only if a CLIP context table is loaded
    context_dim = CONTEXT_TABLE_DIM if CONTEXT_TABLE is not None else None

    if LDM:
        model = UNET(time_steps=NUM_TIME_STEPS,
                    Channels = [256, 256, 512, 1024, 512],
                    Attentions = [False, True, True, False],
                     input_channels=4,
                     output_channels=4,
                     label_embedding=True,
                     use_film=False,
                     dropout_prob=DROPOUT,
                     use_cross_attention=True,
                     context_dim=context_dim,
                     use_flash_attention=use_flash_attention).to(device)
    else:
        model = UNET(time_steps=NUM_TIME_STEPS,
                     input_channels=3,
                     output_channels=3,
                     label_embedding=True,
                     use_film=False,
                     dropout_prob=DROPOUT,
                     use_cross_attention=True,
                     context_dim=context_dim,
                     use_flash_attention=use_flash_attention).to(device)
    
    return model

def _load_clip_context_table(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location='cpu')
    # prefer normalized embeddings
    emb = obj.get('embeddings_norm', obj.get('embeddings'))
    if isinstance(emb, torch.Tensor):
        table = emb.float()
    else:
        table = torch.tensor(emb, dtype=torch.float32)
    
    global CONTEXT_TABLE
    CONTEXT_TABLE = table
    return table

def _labels_to_context(labels: torch.Tensor) -> torch.Tensor:
    assert CONTEXT_TABLE is not None, "CONTEXT_TABLE not loaded"
    # labels: [B] long; table: [11, D] -> [B, D]
    return CONTEXT_TABLE.to(labels.device)[labels.long()]


# ---- Optional CLIP image-side evaluator for debugging label alignment ----
_CLIP_IMG_MODEL = None
_CLIP_IMG_DEVICE = device
_CLIP_IMG_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
_CLIP_IMG_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)

def _ensure_clip_image_encoder(model_name: str = 'ViT-B/32'):
    global _CLIP_IMG_MODEL, _CLIP_IMG_DEVICE
    if _CLIP_IMG_MODEL is not None:
        return _CLIP_IMG_MODEL
    try:
        import clip
    except Exception as e:
        raise RuntimeError("CLIP not installed. Install via: pip install git+https://github.com/openai/CLIP.git") from e
    _CLIP_IMG_MODEL, _ = clip.load(model_name, device=_CLIP_IMG_DEVICE)
    _CLIP_IMG_MODEL.eval()
    for p in _CLIP_IMG_MODEL.parameters():
        p.requires_grad_(False)
    return _CLIP_IMG_MODEL

@torch.no_grad()
def clip_predict_top1_class(image_tensor: torch.Tensor) -> tuple[int, float]:
    """
    Predict top-1 class index (0..9) for a generated image using CLIP's image encoder
    and cosine similarity against the loaded CONTEXT_TABLE text embeddings.

    Expects image_tensor in [-1,1], shape [1,3,H,W] or [3,H,W]. Returns (idx, score).
    Raises if CONTEXT_TABLE not loaded or CLIP not available.
    """
    if CONTEXT_TABLE is None:
        raise RuntimeError("CONTEXT_TABLE not loaded; cannot run CLIP image-side check")

    model = _ensure_clip_image_encoder()

    if image_tensor.ndim == 3:
        img = image_tensor.unsqueeze(0)
    else:
        img = image_tensor
    img = img.to(_CLIP_IMG_DEVICE).float()

    # Map from [-1,1] -> [0,1], resize to 224, normalize with CLIP mean/std
    img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
    img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
    img = (img - _CLIP_IMG_MEAN.to(img.device)) / _CLIP_IMG_STD.to(img.device)

    image_features = model.encode_image(img)
    image_features = torch.nn.functional.normalize(image_features, dim=-1)

    # Compare only against class rows 0..9 (exclude uncond at index 10)
    text_features = CONTEXT_TABLE.to(image_features.device).float()[:10]
    text_features = torch.nn.functional.normalize(text_features, dim=-1)

    sim = (image_features @ text_features.t())  # [1,10]
    val, idx = torch.max(sim, dim=1)
    return int(idx.item()), float(val.item())


class DDPM_beta_t_linear_scheduler(nn.Module):
    def __init__(self, num_steps: int = 1000):
        super().__init__()
        self.beta_t = torch.linspace(1e-4, 0.02, num_steps, requires_grad=False).to(device)
        aplha_t = 1 - self.beta_t
        # alpha_t here stores the cumulative product (alpha_bar)
        self.alpha_t = torch.cumprod(aplha_t, dim=0).requires_grad_(False).to(device)

    def call(self, t):
        return self.alpha_t[t], self.beta_t[t]

class DDPM_beta_t_cosine_scheduler(nn.Module):
    def __init__(self, num_steps = 1000, s = 0.008):
        super().__init__()
        self.T = num_steps
        self.s = s
        self.beta_t, self.alphas, self.alpha_t = self.compute_betas_and_alphas()

    def compute_betas_and_alphas(self):
        
        t = torch.arange(0, self.T + 1).requires_grad_(False).to(device)
        
        # Compute the cumulative noise schedule (alpha_bar) using the cosine function
        alpha_bar = torch.cos(((t / self.T + self.s) / (1 + self.s)) * torch.pi / 2) ** 2
        
        # Compute betas: beta_t = 1 - (alpha_bar[t] / alpha_bar[t-1])
        # Clamp to avoid exact zeros which can cause numerical issues
        betas = torch.clamp(1 - (alpha_bar[1:] / alpha_bar[:-1]), min=1e-4, max=0.999)
        
        # Compute alphas: alpha_t = 1 - beta_t
        alphas = 1 - betas
        
        # Compute the cumulative product of alphas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        
        return betas, alphas, alpha_cumprod
