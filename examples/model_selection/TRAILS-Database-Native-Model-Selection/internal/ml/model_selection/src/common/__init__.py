

import torch

ENABLE_GPU = False

if torch.cuda.is_available():
    ENABLE_GPU = False

