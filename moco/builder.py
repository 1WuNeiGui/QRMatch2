import copy

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """MoCo that shares its query encoder with the main training model.

    The main model is passed to forward as encoder_q. Only an EMA key-encoder
    copy is owned by this module; its parameters are never updated by gradients.
    """

    def __init__(self, encoder_q, dim=512, K=1024, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        # The key encoder is an EMA copy of the main/query encoder.
        # encoder_q is the main classifier model instance, not a constructor.
        self.encoder_k = copy.deepcopy(encoder_q)
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Historical negative images for the FCR branch; kept on CPU.
        self.negative_images = []

    @torch.no_grad()
    def initialize_key_encoder(self, encoder_q):
        """Copy the main encoder parameters into the key encoder (e.g. on resume)."""
        encoder_q = encoder_q.module if hasattr(encoder_q, "module") else encoder_q
        for param_q, param_k in zip(encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.copy_(param_q)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, encoder_q):
        """Update the key encoder as an EMA copy of the shared main encoder."""
        encoder_q = encoder_q.module if hasattr(encoder_q, "module") else encoder_q
        query_params = list(encoder_q.parameters())
        key_params = list(self.encoder_k.parameters())
        if len(query_params) != len(key_params):
            raise RuntimeError("Main encoder and EMA key encoder have different parameter structures.")
        for param_q, param_k in zip(query_params, key_params):
            param_k.data.mul_(self.m).add_(param_q.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, images):
        batch_size = keys.shape[0]
        if self.K % batch_size != 0:
            raise ValueError(
                f"MoCo queue size K={self.K} must be divisible by batch size {batch_size}."
            )

        ptr = int(self.queue_ptr.item())
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

        self.negative_images.extend(
            image.detach().to(device="cpu", dtype=torch.float16).clone()
            for image in images
        )
        if len(self.negative_images) > self.K:
            self.negative_images = self.negative_images[-self.K:]

    def forward(self, encoder_q, im_q, im_k):
        """Compute MoCo logits using the shared main encoder as the query encoder."""
        q = nn.functional.normalize(encoder_q(im_q)["feat"], dim=1)

        # The key branch is a no-gradient EMA copy of the main/query encoder.
        with torch.no_grad():
            self._momentum_update_key_encoder(encoder_q)
            k = nn.functional.normalize(self.encoder_k(im_k)["feat"], dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.detach().clone()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Snapshot history before enqueueing the current positive images, so the
        # paired positive samples are not immediately reused as FCR negatives.
        historical_negative_images = list(self.negative_images)
        self._dequeue_and_enqueue(k, im_k)
        return logits, labels, historical_negative_images


@torch.no_grad()
def concat_all_gather(tensor):
    """Single-process fallback retained for compatibility with the existing code."""
    return tensor.clone()
