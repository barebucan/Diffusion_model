import os
import random
from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_stl10_dataset() -> torchvision.datasets.STL10:
    # Match training transforms: map [0,1] -> [-1, 1]
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])

    trans_flip = transforms.Compose([
        transforms.RandomHorizontalFlip(1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.STL10(root='stl10_data', split='train', download=True, transform=trans_flip)
    dataset_test = torchvision.datasets.STL10(root='stl10_data', split='test', download=True, transform=trans_flip)
    dataset_flip = torchvision.datasets.STL10(root='stl10_data', split='train', download=True, transform=trans_flip)
    dataset_flip_test = torchvision.datasets.STL10(root='stl10_data', split='test', download=True, transform=trans_flip)
    
    combined_dataset = torch.utils.data.ConcatDataset([dataset, dataset_test, dataset_flip, dataset_flip_test])
    return combined_dataset


def get_imagenet_dataset_from_dir(root_dir: str = os.path.join("docs", 'data', 'ImageNET'), image_size: int = 256):
    """
    Loads a folder-per-class dataset from root_dir using ImageFolder.
    Expected layout:
      root_dir/
        class_a/ *.jpg, *.png, ...
        class_b/ *.jpg, *.png, ...

    Images are resized and center-cropped to a square `image_size`, then mapped to [-1, 1].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return torchvision.datasets.ImageFolder(root=root_dir, transform=transform)

def denormalize(x: torch.Tensor) -> torch.Tensor:
    # from [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def reconstruct_with_vae(images: torch.Tensor, vae_id: str, device: torch.device) -> torch.Tensor:
    """
    images: Tensor [N, 3, H, W], value range [-1, 1]
    returns: Tensor [N, 3, H, W], value range [-1, 1]
    """
    try:
        # Newer versions
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL  # type: ignore
    except Exception:
        try:
            # Older layout
            from diffusers.models.autoencoder_kl import AutoencoderKL  # type: ignore
        except Exception:
            try:
                from diffusers import AutoencoderKL  # final fallback
            except Exception as e:
                raise RuntimeError("diffusers is required. Try: pip install -U diffusers") from e

    # Force float32 for broad CPU/GPU compatibility
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float32).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Deterministic: use posterior mean for clean reconstructions
    posterior = vae.encode(images).latent_dist
    latents = posterior.mean
    recon = vae.decode(latents).sample
    return recon

@torch.no_grad()
def encode_dataset(
    dataset: torch.utils.data.Dataset,
    vae_id: str,
    device: torch.device,
    output_path: str,
    batch_size: int = 128
) -> None:
    """
    Encodes the entire dataset using a VAE and saves the latents and labels to a file.
    """
    try:
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    except ImportError:
        try:
            from diffusers import AutoencoderKL
        except ImportError as e:
            raise RuntimeError("diffusers is required. Try: pip install -U diffusers") from e

    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float32).to(device)
    vae.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_latents = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc=f"Encoding dataset with {vae_id}"):
        images = images.to(device)
        posterior = vae.encode(images).latent_dist
        latents = posterior.mean
        all_latents.append(latents.cpu())
        all_labels.append(labels.cpu())

    # Concatenate all batches
    final_latents = torch.cat(all_latents, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    # Create a directory for the output file if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the latents and labels
    torch.save({
        'latents': final_latents,
        'labels': final_labels
    }, output_path)
    print(f"Encoded dataset saved to {output_path}")
    print(f"Latents shape: {final_latents.shape}")
    print(f"Labels shape: {final_labels.shape}")

def save_comparison_grid(originals: List[torch.Tensor], reconstructions: List[torch.Tensor], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(originals)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.1, 4.2))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]
    for i in range(n):
        orig = denormalize(originals[i]).permute(1, 2, 0).cpu().numpy()
        rec = denormalize(reconstructions[i]).permute(1, 2, 0).cpu().numpy()
        axes[0][i].imshow(orig)
        axes[0][i].set_title('orig')
        axes[0][i].axis('off')
        axes[1][i].imshow(rec)
        axes[1][i].set_title('recon')
        axes[1][i].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def enocde_stl10() -> None:
    seed = 42
    vae_id = 'stabilityai/sd-vae-ft-ema'
    random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()
    dataset = get_stl10_dataset()
    encoded_output_path = os.path.join('docs', 'data', 'stl10_latents.pt')
    encode_dataset(dataset = dataset, vae_id = vae_id, device = device, output_path = encoded_output_path)

def enocde_imagenet() -> None:
    seed = 42
    vae_id = 'stabilityai/sd-vae-ft-ema'
    random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()
    dataset = get_imagenet_dataset_from_dir(root_dir=os.path.join("docs", 'data', 'ImageNET'), image_size=256)
    encoded_output_path = os.path.join('docs', 'data', 'imagenet_latents.pt')
    encode_dataset(dataset = dataset, vae_id = vae_id, device = device, output_path = encoded_output_path)

if __name__ == '__main__':
    enocde_imagenet()


