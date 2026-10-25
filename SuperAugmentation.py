import numpy as np
import torch


def Super_Augmentation(image_tensor, mask_prob=0.1, noise_std=1):

    num_channels, height, width = image_tensor.shape

    frame = image_tensor

    mask = np.random.rand(height, width) < mask_prob
    frame = frame.cpu()
    frame = frame * (1 - mask)


    noise = np.random.normal(0, noise_std, size=(num_channels, height, width)).astype(np.float32)
    frame = frame + noise

    frame = np.clip(frame, 0, 255)

    return torch.tensor(frame)