"""Adaptive image augmentation (AIA) selector, following Sec. III-D of the paper.

For every unlabeled sample i the consistency loss (strong view against the weak-view hard
pseudo-label) is smoothed by an EMA (Eq. 13/14):
        H^t(i) = (1 - alpha) * H^{t-1}(i) + alpha * l^t(i)
Samples are grouped by their predicted class c_i = argmax p_w. For every class a threshold
tau_c is computed with Otsu's method on the EMA losses of that class (Eq. 15). A sample is
"simple" when H(i) < tau_{c_i} (Eq. 16). Only simple samples are re-augmented.

Pure numpy, so it can be tested without a GPU. Also keeps the statistics needed for the
analysis of the selected set (precision against the ground truth, thresholds, trajectories).
"""
import csv
import os

import numpy as np


def otsu_threshold(values, bins=256):
    """Otsu threshold of 1-D float values (histogram with `bins` bins between min and max).
    Returns None when the values cannot be split (fewer than 2 points or all equal)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2 or v.max() - v.min() < 1e-12:
        return None
    hist, edges = np.histogram(v, bins=bins)
    p = hist.astype(np.float64) / hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    w1 = np.cumsum(p)
    w2 = 1.0 - w1
    m1 = np.cumsum(p * centers)
    mt = m1[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        mean1 = m1 / w1
        mean2 = (mt - m1) / w2
        var = w1 * w2 * (mean1 - mean2) ** 2
    var[~np.isfinite(var)] = -1.0
    k = int(np.argmax(var))
    return float(edges[k + 1])  # upper edge of the best lower bin


class AIASelector:
    def __init__(self, n_samples, num_classes, labels, alpha=0.5, out_dir=None,
                 track_n=8, seed=0):
        self.n = n_samples
        self.num_classes = num_classes
        self.alpha = alpha
        self.labels = np.asarray(labels, dtype=np.int64)
        self.H = np.full(n_samples, np.nan)        # EMA consistency loss
        self.last_loss = np.full(n_samples, np.nan)  # raw loss at the latest visit
        self.cls = np.full(n_samples, -1, dtype=np.int64)  # latest predicted class
        self.tau = np.full(num_classes, -np.inf)   # class-wise thresholds
        self.out_dir = out_dir
        rng = np.random.RandomState(seed)
        self.tracked = rng.choice(n_samples, size=min(track_n, n_samples), replace=False)
        self.stat_rows, self.traj_rows = [], []

    # ---- per-iteration ----
    def update(self, idx, losses, pred):
        idx = np.asarray(idx)
        losses = np.asarray(losses, dtype=np.float64)
        old = self.H[idx]
        self.H[idx] = np.where(np.isnan(old), losses,
                               (1.0 - self.alpha) * old + self.alpha * losses)
        self.last_loss[idx] = losses
        self.cls[idx] = np.asarray(pred)

    def is_simple(self, idx):
        """Boolean array for a batch of indices, using the state before the current update."""
        idx = np.asarray(idx)
        c = self.cls[idx]
        h = self.H[idx]
        ok = (c >= 0) & np.isfinite(h)
        tau = np.where(c >= 0, self.tau[np.clip(c, 0, None)], -np.inf)
        return ok & (h < tau)

    def correct(self, idx):
        idx = np.asarray(idx)
        return self.cls[idx] == self.labels[idx]

    # ---- per-epoch ----
    def recompute_thresholds(self):
        for c in range(self.num_classes):
            m = (self.cls == c) & np.isfinite(self.H)
            t = otsu_threshold(self.H[m])
            self.tau[c] = -np.inf if t is None else t

    def end_epoch(self, epoch):
        """Recompute thresholds, log statistics of the simple set and tracked trajectories."""
        self.recompute_thresholds()
        seen = (self.cls >= 0) & np.isfinite(self.H)
        simple = np.zeros(self.n, dtype=bool)
        all_idx = np.arange(self.n)
        simple[seen] = self.is_simple(all_idx[seen])
        right = self.cls == self.labels
        n_seen, n_simple = int(seen.sum()), int(simple.sum())
        n_simple_ok = int((simple & right).sum())
        row = {
            'epoch': epoch + 1, 'n_seen': n_seen,
            'pl_acc_seen': float(right[seen].mean()) if n_seen else float('nan'),
            'n_simple': n_simple, 'n_simple_correct': n_simple_ok,
            'n_simple_wrong': n_simple - n_simple_ok,
            'simple_precision': n_simple_ok / n_simple if n_simple else float('nan'),
        }
        for c in range(self.num_classes):
            row['tau_c%d' % c] = float(self.tau[c])
        self.stat_rows.append(row)
        for i in self.tracked:
            self.traj_rows.append({
                'epoch': epoch + 1, 'index': int(i), 'true': int(self.labels[i]),
                'pred': int(self.cls[i]), 'loss': self.last_loss[i], 'ema': self.H[i],
                'tau': float(self.tau[self.cls[i]]) if self.cls[i] >= 0 else float('nan'),
                'simple': int(simple[i])})
        self.flush()
        return row

    def flush(self):
        if not self.out_dir:
            return
        for name, rows in (('aia_stats.csv', self.stat_rows), ('aia_traj.csv', self.traj_rows)):
            if rows:
                with open(os.path.join(self.out_dir, name), 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
