"""Plot the AIA analysis from <run>/aia_stats.csv and <run>/aia_traj.csv.

    python plot_aia_stats.py --run result_aia --out figures/aia

Outputs (svg + png):
    <out>_precision.*  precision of the simple set (share of simple samples whose pseudo-label is
                       correct) and pseudo-label accuracy of all seen samples, per epoch
    <out>_tau.*        class-wise Otsu thresholds per epoch (one thin line per class, mean in bold)
    <out>_traj.*       raw / EMA consistency loss of the tracked samples with the threshold of
                       their class (dashed); one panel per sample, title shows true/pred class
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral"]
plt.rcParams["svg.fonttype"] = "path"
FS, FRAME = 14, "#999999"


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def box(ax):
    for s in ax.spines.values():
        s.set_color(FRAME)
    ax.grid(True, linestyle="--", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", color=FRAME, labelsize=FS - 1)


def save(fig, out):
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out + ".svg")
    fig.savefig(out + ".png", dpi=300)
    print("Saved", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="figures/aia")
    p.add_argument("--aia-start", type=int, default=100)
    a = p.parse_args()

    st = read(os.path.join(a.run, "aia_stats.csv"))
    ep = np.array([float(r["epoch"]) for r in st])
    f = lambda k: np.array([float(r[k]) for r in st])

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ep, f("simple_precision") * 100, color="#4E9AF1", lw=1.8, label="Simple set")
    ax.plot(ep, f("pl_acc_seen") * 100, color="#6CC57C", lw=1.8, label="All seen samples")
    ax.axvline(a.aia_start + 1, color="#999999", ls="--", lw=1)
    ax.set_xlim(ep.min(), ep.max()); ax.set_xlabel("Epoch", fontsize=FS)
    ax.set_ylabel("Pseudo-label accuracy (%)", fontsize=FS)
    box(ax); ax.legend(frameon=False, fontsize=FS)
    save(fig, a.out + "_precision")

    tau_keys = [k for k in st[0] if k.startswith("tau_c")]
    T = np.stack([f(k) for k in tau_keys])
    T[~np.isfinite(T)] = np.nan
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for row in T:
        ax.plot(ep, row, color="#4E9AF1", lw=0.8, alpha=0.45)
    ax.plot(ep, np.nanmean(T, axis=0), color="#D62728", lw=2, label="Mean over classes")
    ax.axvline(a.aia_start + 1, color="#999999", ls="--", lw=1)
    ax.set_xlim(ep.min(), ep.max()); ax.set_xlabel("Epoch", fontsize=FS)
    ax.set_ylabel("Otsu threshold", fontsize=FS)
    box(ax); ax.legend(frameon=False, fontsize=FS)
    save(fig, a.out + "_tau")

    tr = read(os.path.join(a.run, "aia_traj.csv"))
    ids = sorted({int(r["index"]) for r in tr})
    cols = 4 if len(ids) > 4 else len(ids)
    rows = int(np.ceil(len(ids) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), squeeze=False)
    for ax, i in zip(axs.ravel(), ids):
        r = [x for x in tr if int(x["index"]) == i]
        e = np.array([float(x["epoch"]) for x in r])
        ax.plot(e, [float(x["loss"]) for x in r], color="#BBBBBB", lw=0.8, label="Loss")
        ax.plot(e, [float(x["ema"]) for x in r], color="#4E9AF1", lw=1.6, label="EMA")
        ax.plot(e, [float(x["tau"]) for x in r], color="#D62728", lw=1.2, ls="--", label="Threshold")
        ax.set_title("true %s, pred %s" % (r[-1]["true"], r[-1]["pred"]), fontsize=FS - 3)
        ax.set_xlabel("Epoch", fontsize=FS - 3); box(ax)
    axs.ravel()[0].legend(frameon=False, fontsize=FS - 5)
    for ax in axs.ravel()[len(ids):]:
        ax.axis("off")
    save(fig, a.out + "_traj")


if __name__ == "__main__":
    main()
