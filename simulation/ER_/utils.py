

import torch


def device():
    if torch.cuda.is_available():
        print("Nvidia GPU Detected")
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     print("Apple Silicon Detected")
    #     return torch.device("mps")
    else:
        print("Use CPU Only")
        return torch.device("cpu")