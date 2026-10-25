"""Gradient conflict analysis between the DDCL branch and the consistency branch.

For a measured iteration, the gradients of L_DDCL and L_cons with respect to the
shared encoder parameters are computed separately, and their cosine similarity is

    CosSim = g_DDCL^T g_cons / (||g_DDCL||_2 * ||g_cons||_2 + eps)

CosSim > 0: the two gradients point in a similar direction.
CosSim < 0: the two gradients conflict.
CosSim ~ 0: the two gradients are nearly orthogonal.

Raw per-iteration values and per-epoch summaries are written to CSV files so the
curve can be re-plotted later with plot_grad_conflict.py.
"""

import csv
import math
import os

import torch


ITER_FIELDS = ["epoch", "iter", "cos_sim", "norm_ddcl", "norm_cons", "loss_ddcl", "loss_cons"]
EPOCH_FIELDS = ["epoch", "cos_mean", "cos_std", "num_valid", "num_total"]


def should_measure(epoch_1based, start, interval):
    """True for epochs start, start + interval, start + 2 * interval, ..."""
    if interval <= 0 or epoch_1based < start:
        return False
    return (epoch_1based - start) % interval == 0


def encoder_parameters(model):
    """Shared encoder parameters theta (the classifier head fc is excluded)."""
    model = model.module if hasattr(model, "module") else model
    return [p for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith("fc.")]


def _flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = [torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
            for g, p in zip(grads, params)]
    return torch.cat(flat).float()


def gradient_cosine(loss_ddcl, loss_cons, params, eps=1e-12):
    """Return (cos_sim, ||g_DDCL||, ||g_cons||). cos_sim is NaN if a gradient is zero.

    retain_graph=True keeps the graph, so loss.backward() can still be called afterwards.
    """
    g_ddcl = _flat_grad(loss_ddcl, params)
    g_cons = _flat_grad(loss_cons, params)
    n_ddcl = g_ddcl.norm(p=2).item()
    n_cons = g_cons.norm(p=2).item()
    if n_ddcl == 0.0 or n_cons == 0.0:
        cos = float("nan")
    else:
        cos = (torch.dot(g_ddcl, g_cons) / (n_ddcl * n_cons + eps)).item()
    del g_ddcl, g_cons
    return cos, n_ddcl, n_cons


class GradConflictLogger:
    """Collects per-iteration cosine values and writes them to CSV files in out_dir."""

    def __init__(self, out_dir, tag="with_dtf"):
        os.makedirs(out_dir, exist_ok=True)
        self.iter_path = os.path.join(out_dir, f"grad_conflict_{tag}_iters.csv")
        self.epoch_path = os.path.join(out_dir, f"grad_conflict_{tag}_epochs.csv")
        self._init_file(self.iter_path, ITER_FIELDS)
        self._init_file(self.epoch_path, EPOCH_FIELDS)
        self._current = []

    @staticmethod
    def _init_file(path, fields):
        # Keep existing rows (e.g. when resuming); write the header only for a new file.
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def log_iter(self, epoch, it, cos, n_ddcl, n_cons, loss_ddcl, loss_cons):
        row = [epoch, it, cos, n_ddcl, n_cons, loss_ddcl, loss_cons]
        with open(self.iter_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self._current.append(cos)

    def end_epoch(self, epoch):
        """Write the epoch mean/std over valid iterations. Returns the mean (or None)."""
        if not self._current:
            return None
        valid = [c for c in self._current if not math.isnan(c)]
        if valid:
            mean = sum(valid) / len(valid)
            std = math.sqrt(sum((c - mean) ** 2 for c in valid) / len(valid))
        else:
            mean, std = float("nan"), float("nan")
        with open(self.epoch_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean, std, len(valid), len(self._current)])
        self._current = []
        return None if math.isnan(mean) else mean
