import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
# This next line is the most important one!
print(f"PyTorch Build: {torch.version.cuda}")