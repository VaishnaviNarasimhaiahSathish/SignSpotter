import random
import numpy as np
import torch

# ---------------------------------------------------
# Reproducibility
# ---------------------------------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------
# Unnormalize image for plotting
# ---------------------------------------------------

def unnormalize(img):
    img = img.numpy().transpose(1, 2, 0)
    img = (img * 0.5) + 0.5
    return np.clip(img, 0, 1)

