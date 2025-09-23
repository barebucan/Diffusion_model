import torch
import torch.nn as nn
from typing import Optional

VAE_MODEL: Optional[nn.Module] = None
VAE_ID = 'stabilityai/sd-vae-ft-ema'

# This helper function is great. We'll keep it as a private function
# within this module.
def _import_autoencoder_kl():
    """Robust import for AutoencoderKL across diffusers versions."""
    try:
        # Newer diffusers
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        return AutoencoderKL
    except ImportError:
        # Fallback for older diffusers
        from diffusers import AutoencoderKL
        return AutoencoderKL

class VAEWrapper:
    """
    A wrapper class for the Hugging Face AutoencoderKL model.

    This class encapsulates the VAE model, its device, and provides clean,
    self-contained methods for encoding and decoding, correctly handling the
    scaling factor.
    """
    def __init__(self, vae_id: str, device: torch.device, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        
        # --- Logic from `load_vae()` is now in `__init__` ---
        AutoencoderKL = _import_autoencoder_kl()
        print(f"Loading VAE model: {vae_id} to {self.device}")
        model = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype).to(self.device)
        model.eval()
        model.requires_grad_(False) # More concise way to freeze
        self.model = model
        
        # This scaling factor is crucial and was missing from your decode function
        self.scaling_factor = self.model.config.scaling_factor

    @torch.no_grad()
    def encode(self, images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Encode images from image space [-1, 1] to latent space.
        
        Args:
            images (torch.Tensor): Input images tensor [N, 3, H, W] on any device.
            deterministic (bool): If True, use the posterior mean. Otherwise, sample.

        Returns:
            torch.Tensor: Latent representations [N, 4, H/8, W/8] scaled by the factor.
        """
        # --- Logic from `vae_encode()` is now a method ---
        x = images.to(self.device, dtype=self.dtype)
        posterior = self.model.encode(x).latent_dist
        
        if deterministic:
            latents = posterior.mean
        else:
            latents = posterior.sample()
            
        # The latents must be scaled by this factor
        return latents * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents from latent space back to image space [-1, 1].

        Args:
            latents (torch.Tensor): Latent representations [N, 4, H/8, W/8] on any device.

        Returns:
            torch.Tensor: Decoded images tensor [N, 3, H, W] in [-1, 1].
        """
        # --- Logic from `vae_decode()` is now a method, WITH THE FIX ---
        # The latents must be un-scaled before decoding
        latents_unscaled = latents / self.scaling_factor
        
        z = latents_unscaled.to(self.device, dtype=self.dtype)
        return self.model.decode(z).sample

# --- SINGLE, SHARED INSTANCE ---
# This pattern provides a clean way to have a single VAE instance
# available across your project, but without using messy global variables.

vae: Optional[VAEWrapper] = None

def get_vae() -> VAEWrapper:
    """Initializes and returns the shared VAE instance."""
    global vae
    if vae is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae = VAEWrapper(vae_id=VAE_ID, device=device)
    return vae