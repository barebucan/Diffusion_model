import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from utils import load_checkpoint
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from utils import _cfg_predict_eps_from_v
import torch
import config
from PIL import Image
import utils
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from visualization import sample_noise, run_reverse_process, tensor_to_image, sample_dpm_solver
def compute_spectrum(img):
    # img: [3, H, W], range [-1,1]
    fft = torch.fft.fft2(img)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shift)
    return magnitude.mean(dim=0)  # average channels

def radial_profile(magnitude):
    H, W = magnitude.shape
    y, x = np.indices((H, W))
    center = np.array([H // 2, W // 2])
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), magnitude.cpu().numpy().ravel())
    nr = np.bincount(r.ravel())
    radial = tbin / (nr + 1e-8)
    return radial

def test_spectrum(real_imgs, fake_imgs):
    real_freq = np.mean([radial_profile(compute_spectrum(img)) for img in real_imgs], axis=0)
    fake_freq = np.mean([radial_profile(compute_spectrum(img)) for img in fake_imgs], axis=0)

    plt.loglog(real_freq, label='Real')
    plt.loglog(fake_freq, label='Generated')
    plt.legend()
    plt.title('Power Spectrum')
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Power')

def test_spectrum_for_each_folder(real_imgs_folders, fake_imgs_folders, titles ):
    fake_freqs = []
    real_freqs = []
    for real_imgs, fake_imgs in zip(real_imgs_folders, fake_imgs_folders):
        real_freq = np.mean([radial_profile(compute_spectrum(img)) for img in real_imgs], axis=0)
        fake_freq = np.mean([radial_profile(compute_spectrum(img)) for img in fake_imgs], axis=0)
        real_freqs.append(real_freq)
        fake_freqs.append(fake_freq)
    
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = axes.flatten() 
    for i, (real_freq, fake_freq) in enumerate(zip(real_freqs, fake_freqs)):
        axes[i].loglog(real_freq, label='Real')
        axes[i].loglog(fake_freq, label='Generated')
        axes[i].legend()
        axes[i].set_title(f'{titles[i]}')
        axes[i].set_xlabel('Spatial Frequency')
        axes[i].set_ylabel('Power')
    plt.tight_layout()
    plt.savefig('spectrum_for_each_folder.png')
    plt.close()

def create_images_for_analysis(model, scheduler):
    
    save_dir = "intermiate imagees/generated"
    os.makedirs(save_dir, exist_ok=True)
    titles = []
    fake_imgs_folders = []
    for cls in config.IMAGENET_ID_TO_NAME_WITHOUT_UNCOND.keys():
        fake_imgs = []    
        for i in range(config.NUM_IMAGES_PER_FOLDER_FOR_SPECTRUM):
            with torch.no_grad():
                latents = sample_dpm_solver(
                    model,
                    scheduler,
                    cls,
                    cfg_scale = 6,
                    num_inference_steps = config.NUM_TIME_STEPS_DPM,
                    device = device,
                    )
                
                img_np = tensor_to_image(latents)

                try:
                    # Reconstruct tensor in [-1,1] for CLIP
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                except Exception:
                    pass
                fake_imgs.append(img_tensor)
        fake_imgs_folders.append(fake_imgs)
        titles.append(config.IMAGENET_ID_TO_NAME_WITHOUT_UNCOND[cls])
    print("Fake images folders created")
    real_imgs_folders = []
    real_imgs_path_folder = "docs\data\ImageNET"
    for cls in config.IMAGENET_ID_TO_NAME_WITHOUT_UNCOND.keys():
        real_imgs = []
        title = config.IMAGENET_ID_TO_NAME[cls]
        title = title.replace(" ", "_")
        for i,img_path in enumerate(os.listdir(os.path.join(real_imgs_path_folder, title))):
            if i == config.NUM_IMAGES_PER_FOLDER_FOR_SPECTRUM:
                break
            img = Image.open(os.path.join(real_imgs_path_folder, title, img_path))
            img = transforms.ToTensor()(img)
            real_imgs.append(img)
        real_imgs_folders.append(real_imgs)

        
    print("Real images folders created")
    test_spectrum_for_each_folder(real_imgs_folders, fake_imgs_folders, titles)

def main():
    checkpoint_path = 'stl10_checkpoint_epoch_55.pt'

    model, ema = load_checkpoint(checkpoint_path)
    model = ema.module.eval()

    from diffusers import DPMSolverMultistepScheduler
    
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=config.NUM_TIME_STEPS,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",  # Match your training
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    
    create_images_for_analysis(model, scheduler)

if __name__ == '__main__':
    main()