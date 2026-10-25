import torch
import torch.nn as nn


class FCR(nn.Module):
    """Frequency Contrastive Regularization over encoder feature maps.

    Inputs are feature maps produced by the trainable encoder, not raw images.
    The query feature map remains attached to autograd so FCR gradients can
    update the shared encoder. Positive/negative feature maps may be detached
    key representations.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, anchor_features, positive_features, negative_features=None):
        if anchor_features.ndim != 4 or positive_features.ndim != 4:
            raise ValueError("FCR expects feature maps shaped [B, C, H, W].")
        if anchor_features.shape != positive_features.shape:
            raise ValueError("Anchor and positive feature maps must have the same shape.")

        # A zero loss connected to anchor_features safely handles queue warm-up.
        if negative_features is None or negative_features.shape[0] == 0:
            return anchor_features.sum() * 0.0
        if negative_features.ndim != 4 or negative_features.shape[1:] != anchor_features.shape[1:]:
            raise ValueError("Negative feature maps must match anchor C/H/W dimensions.")

        # FFT is applied to spatial feature maps; abs() makes distances real-valued.
        anchor_fft = torch.fft.fft2(anchor_features, dim=(-2, -1))
        positive_fft = torch.fft.fft2(positive_features, dim=(-2, -1))
        negative_fft = torch.fft.fft2(negative_features, dim=(-2, -1))

        batch_size = anchor_fft.shape[0]
        losses = []

        # Compare every query against every negative feature map passed to FCR.
        # QRMatch.py chunks the 1024-entry queue for feature extraction to control
        # peak memory, then combines chunk losses weighted by their negative count.
        for i in range(batch_size):
            d_pos = (anchor_fft[i] - positive_fft[i]).abs().mean()
            d_neg = (
                anchor_fft[i].unsqueeze(0) - negative_fft
            ).abs().mean(dim=(1, 2, 3))
            losses.append((d_pos / (d_neg + self.eps)).mean())

        return torch.stack(losses).mean() if losses else anchor_features.sum() * 0.0
