import numpy as np
import torch

# Valid range of an ImageNet-normalized pixel, i.e. (0 - mean) / std and (1 - mean) / std.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_LOW, _HIGH = (0 - _MEAN) / _STD, (1 - _MEAN) / _STD


def Super_Augmentation(image_tensor, mask_prob=0.1, noise_std=1, clip=True):
    """A(x) = M(G(x)) (paper Sec. III-D): Gaussian noise first, then a random pixel mask.

    Input is an ImageNet-normalized (C, H, W) tensor. The mask zeroes a pixel in all channels
    (zero is the dataset mean colour in normalized space). With clip=True the noisy image is
    kept inside the valid normalized pixel range; the old code clipped to [0, 255], which
    wrongly removed every negative normalized value.
    """
    num_channels, height, width = image_tensor.shape
    frame = image_tensor.detach().cpu().numpy().astype(np.float32)

    noise = np.random.normal(0, noise_std, size=(num_channels, height, width)).astype(np.float32)
    frame = frame + noise                                   # G(x)
    if clip and num_channels == 3:
        frame = np.clip(frame, _LOW, _HIGH)

    mask = np.random.rand(height, width) < mask_prob        # M(.)
    frame = frame * (1 - mask).astype(np.float32)

    return torch.from_numpy(frame)
