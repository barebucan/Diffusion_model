import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
from unet import UNET
import os
from tqdm import tqdm
from timm.utils import ModelEmaV2

NUM_EPOCHS = 50
BATCH_SIZE = 64
LR = 5e-5
EMA_DECAY = 0.999
NUM_TIME_STEPS = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory for saving intermediate images
OUTPUT_DIR = 'intermiate imagees'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRANS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class DDPM_beta_t_linear_scheduler(nn.Module):
    def __init__(self, num_steps = 1000):
        super().__init__()
        self.beta_t = torch.linspace(1e-4, 0.02, num_steps, requires_grad=False).to(device)
        aplha_t = 1 - self.beta_t
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
        betas = torch.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 0, 0.999)
        
        # Compute alphas: alpha_t = 1 - beta_t
        alphas = 1 - betas
        
        # Compute the cumulative product of alphas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        
        return betas, alphas, alpha_cumprod

def save_denoising_collage(images, timesteps, filename):
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 1.5, 1.5))
    if cols == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        img = images[idx]
        # Ensure tensor on CPU and in 2D (H, W)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        if img.ndim == 4:  # [B, C, H, W]
            img = img[0, 0]
        elif img.ndim == 3:  # [C, H, W]
            img = img[0]
        else:  # [H, W] or others
            img = img.squeeze()
        ax.imshow(img.numpy(), cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(f"t={timesteps[idx]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()


def generate_denoise_collage(model, scheduler, epoch, step, timesteps=None):
    """Generate a collage of intermediate denoising steps at selected timesteps."""
    if timesteps is None:
        timesteps = [999, 700, 550, 400, 300, 200, 100, 50, 15, 0]

    with torch.no_grad():
        model.eval()
        random_label = torch.randint(0, 10, (1,)).to(device)
        z = torch.randn(1, 1, 32, 32).to(device)

        snapshots = {}
        for t in reversed(range(1, NUM_TIME_STEPS)):
            t_tensor = torch.tensor([t]).to(device)
            predicted_noise = model(z, t_tensor, random_label)

            alpha_bar_t = scheduler.alpha_t[t]
            beta_t = scheduler.beta_t[t]
            alpha_t = 1.0 - beta_t
            alpha_bar_prev = scheduler.alpha_t[t-1] if t > 0 else torch.tensor(1.0, device=device)

            # DDPM sampling mean
            z = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)

            # Add noise with posterior variance for t>1 (final step is deterministic)
            if t > 1:
                noise = torch.randn_like(z)
                posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                z = z + torch.sqrt(posterior_var) * noise

            if t in timesteps:
                snap = z[:, :, 2:-2, 2:-2]  # crop to 28x28 to match training normalization
                snap = torch.clamp((snap + 1) / 2, 0, 1)
                snapshots[t] = snap

        # Final step t=0
        t_tensor = torch.tensor([0]).to(device)
        predicted_noise = model(z, t_tensor, random_label)
        alpha_bar_0 = scheduler.alpha_t[0]
        beta_0 = scheduler.beta_t[0]
        alpha_0 = 1.0 - beta_0
        final_image = (1.0 / torch.sqrt(alpha_0)) * (z - (beta_0 / torch.sqrt(1.0 - alpha_bar_0)) * predicted_noise)
        final_image = final_image[:, :, 2:-2, 2:-2]
        final_image = torch.clamp((final_image + 1) / 2, 0, 1)
        snapshots[0] = final_image

        ordered_images = [snapshots[t] for t in timesteps if t in snapshots]
        if len(ordered_images) == 0:
            return
        out_path = f'epoch_{epoch}_step_{step}_collage.png'
        save_denoising_collage(ordered_images, [t for t in timesteps if t in snapshots], out_path)

def train(checkpoint_path = None):
    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)

    train_data =  torchvision.datasets.MNIST(root='mnist_data', train=True, transform=TRANS)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    model = UNET(time_steps=NUM_TIME_STEPS, label_embedding=True,
                 Attentions=[False, False, False, False, False, False], use_film=False)
    model = model.to(device)
    
    ema = ModelEmaV2(model, decay = EMA_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction='mean')

    infer_interval = max(1, len(train_loader)//4)
    collage_timesteps = [999, 700, 550, 400, 300, 200, 100, 50, 15, 0]
 
    for epoch in range(NUM_EPOCHS):
        sum_loss = 0
        for i, (x, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            x = x.to(device)
            labels = labels.to(device)

            # Classifier-free guidance: randomly drop labels (set to unconditional)
            mask = torch.rand(x.shape[0], device=device)
            labels[mask < 0.1] = 10  # Use label 10 for unconditional (null class)
            labels = labels.to(device)

            t = torch.randint(0, NUM_TIME_STEPS, (x.shape[0],), requires_grad=False)
            a = scheduler.alpha_t[t].to(device).view(-1, 1, 1, 1)
            x = F.pad(x, (2,2,2,2))
            e = torch.randn_like(x, requires_grad=False).to(device)
            optimizer.zero_grad()
            x = (torch.sqrt(a) * x + torch.sqrt(1-a) * e)

            output = model(x, t, labels)
            loss = criterion(output, e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            sum_loss += loss.item()
            optimizer.step()
            ema.update(model)

            # Mid-epoch inference collage
            if (i + 1) % infer_interval == 0:
                generate_denoise_collage(ema.module, scheduler, epoch+1, i+1, collage_timesteps)
        print(f'Epoch {epoch+1} | Loss {sum_loss / len(train_loader):.5f}')
        
        # Generate and save an image of a random number after each epoch
        generate_random_number_image(ema.module, scheduler, epoch+1)

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    #torch.save(checkpoint, 'checkpoint.pt')
def display(images):

    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = x.permute(1, 2, 0)  # Rearrange to (height, width, channels)
        x = x.cpu().numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, "number_transform.png"))

def generate_random_number_image(model, scheduler, epoch):
    """Generate and save an image of a random number after each epoch"""
    with torch.no_grad():
        model.eval()
        
        # Generate a random label (0-9 for MNIST digits)
        random_label = torch.randint(0, 10, (1,)).to(device)
        
        # Start with pure noise
        z = torch.randn(1, 1, 32, 32).to(device)
        
        # Denoising process
        for t in reversed(range(1, NUM_TIME_STEPS)):
            t_tensor = torch.tensor([t]).to(device)

            predicted_noise = model(z, t_tensor, random_label)

            alpha_bar_t = scheduler.alpha_t[t]
            beta_t = scheduler.beta_t[t]
            alpha_t = 1.0 - beta_t
            alpha_bar_prev = scheduler.alpha_t[t-1] if t > 0 else torch.tensor(1.0, device=device)

            # DDPM sampling mean
            z = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)

            # Add noise with posterior variance for t>1 (final step is deterministic)
            if t > 1:
                noise = torch.randn_like(z)
                posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                z = z + torch.sqrt(posterior_var) * noise
        
        # Final step
        t_tensor = torch.tensor([0]).to(device)
        predicted_noise = model(z, t_tensor, random_label)
        alpha_bar_0 = scheduler.alpha_t[0]
        beta_0 = scheduler.beta_t[0]
        alpha_0 = 1.0 - beta_0
        final_image = (1.0 / torch.sqrt(alpha_0)) * (z - (beta_0 / torch.sqrt(1.0 - alpha_bar_0)) * predicted_noise)
        
        # Remove padding and save image
        final_image = final_image[:, :, 2:-2, 2:-2]  # Remove the padding added during training
        final_image = torch.clamp((final_image + 1) / 2, 0, 1)  # Clamp to valid range
        
        # Convert to displayable format
        img_np = final_image.squeeze().cpu().numpy()
        
        # Save the generated image
        plt.figure(figsize=(3, 3))
        plt.imshow(img_np, cmap='gray')
        plt.title(f'Generated digit {random_label.item()} - Epoch {epoch}')
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, f'epoch_{epoch}_digit_{random_label.item()}.png'), bbox_inches='tight', dpi=150)
        plt.close()  # Close the figure to free memory
        
        print(f'Generated and saved image of digit {random_label.item()} for epoch {epoch}')

def inference(checkpoint_path):

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNET(Attentions=[False, False, False, False, False, False], use_film=False).to(device)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV2(model, decay=EMA_DECAY)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_beta_t_cosine_scheduler(NUM_TIME_STEPS)
    times = [0,15,50,100,200,300,400,550,700,999]
    images = []
    train_data =  torchvision.datasets.MNIST(root='mnist_data', train=True, transform=transforms.ToTensor())

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            
            z = train_data[i][0].unsqueeze(0).to(device)
            z = F.pad(z, (2,2,2,2))
            label_z = train_data[i][1]
            # z = torch.randn(1, 1, 32, 32).to(device)
            label = torch.randint(0, 10, (1,)).to(device)
            while label_z == label:
                label = torch.randint(0, 10, (1,)).to(device)
            print("Generating image of ", label.item(), " from image of ", label_z)
            for t in reversed(range(1, NUM_TIME_STEPS)):
                t = [t]
                temp = (scheduler.beta_t[t]/( (torch.sqrt(1-scheduler.alpha_t[t]))*(torch.sqrt(1-scheduler.beta_t[t])) ))
                z = (1/(torch.sqrt(1-scheduler.beta_t[t])))*z - (temp*model(z.to(device),t, label))
                if t[0] in times:
                    images.append(z)
                e = torch.randn(1, 1, 32, 32).to(device)
                z = z + (e*torch.sqrt(scheduler.beta_t[t]))
            temp = scheduler.beta_t[0]/( (torch.sqrt(1-scheduler.alpha_t[0]))*(torch.sqrt(1-scheduler.beta_t[0])) )
            x = (1/(torch.sqrt(1-scheduler.beta_t[0])))*z - (temp*model(z.to(device),[0], label))

            images.append(x)
            display(images)
            images = []

def main():
    inference('stl10_checkpoint_epoch_2000.pt')


if __name__ == '__main__':
    main()


