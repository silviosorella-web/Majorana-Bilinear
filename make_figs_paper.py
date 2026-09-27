"""Vector figures for the draft (print, light background).  Colors: dataviz reference slots 1-2
(validated); identity also carried by line style and markers so the figures survive grayscale."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt
from sorella08_check import GF

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
                     "axes.linewidth": 0.8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False})
INK2, GRID, C1, C2, C3 = "#505967", "#e2e5ea", "#2a78d6", "#eb6834", "#1baf7a"
OUT = "../paper/figs/"

# ---- Fig. 1: compact seeds at contact d = a ----
a = np.concatenate([np.geomspace(1e-3, 0.02, 12), np.linspace(0.025, 0.55, 80)])
B8, Bo = [], []
for x in a:
    G, F, Tzz = GF(x, x)
    B8.append(2 * sqrt(2) * G * F); Bo.append(2 * sqrt(Tzz**2 + (G * F)**2))
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.grid(True, color=GRID, lw=0.6)
ax.axhline(2 * sqrt(2), color=INK2, lw=0.8, ls=(0, (4, 3)))
ax.axhline(2, color=INK2, lw=0.8)
ax.text(0.545, 2 * sqrt(2) + 0.035, r"$2\sqrt{2}$", color=INK2, ha="right", fontsize=9)
ax.text(0.545, 2.03, r"$2$", color=INK2, ha="right", fontsize=9)
ax.plot(a, B8, color=C1, lw=1.6, label=r"$2\sqrt{2}\,G\,F$")
ax.plot(a, Bo, color=C2, lw=1.6, ls=(0, (6, 2)), label=r"$2\sqrt{T_{zz}^2+G^2F^2}$")
for ac, col in [(0.17361, C1), (0.40829, C2)]:
    ax.plot([ac], [2], marker="o", ms=5, color=col, mec="white", mew=1.0, zorder=4)
ax.annotate(r"$a_c\simeq0.174$", (0.17361, 2), xytext=(-62, -15), textcoords="offset points", fontsize=8.5)
ax.annotate(r"$a_c\simeq0.408$", (0.40829, 2), xytext=(-18, 8), textcoords="offset points", fontsize=8.5)
ax.set_xlim(0, 0.55); ax.set_ylim(1.2, 2.95)
ax.set_xlabel(r"$a$"); ax.set_ylabel(r"$\mathcal{B}_{\rm CHSH}$")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig_compact_seeds.pdf"); plt.close(fig)

# ---- Fig. 2: Gaussian seeds centred at omega = 0 ----
sig = np.array([1e-2, 5e-3, 2e-3, 1e-3])
opt6 = np.array([1.37671e-3, 3.44435e-4, 5.51212e-5, 1.37807e-5])
expl = np.array([1.67029e-3, 4.17800e-4, 6.68580e-5, 1.67150e-5])
Ns = np.array([3, 4, 6, 8, 10]); cN = np.array([20.936, 20.936, 13.781, 10.359, 8.320])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax1.grid(True, color=GRID, lw=0.6)
ss = np.geomspace(8e-4, 1.2e-2, 50)
ax1.plot(ss, 13.78 * ss**2, color=C2, lw=0.9, ls=(0, (4, 3)))
ax1.plot(ss, 16.71 * ss**2, color=C1, lw=0.9, ls=(0, (1, 1.5)))
ax1.plot(sig, opt6, ls="none", marker="s", ms=5, color=C2, mec="white", mew=0.8, label=r"optimum, $N=6$")
ax1.plot(sig, expl, ls="none", marker="^", ms=5.5, color=C1, mec="white", mew=0.8, label="explicit configuration")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel(r"$\sigma$"); ax1.set_ylabel(r"$2\sqrt{2}-\mathcal{B}_{\rm CHSH}$")
ax1.legend(loc="upper left", fontsize=8.5)
ax1.text(0.95, 0.04, "(a)", transform=ax1.transAxes, fontsize=9, ha="right")
ax2.grid(True, color=GRID, lw=0.6)
ax2.plot(Ns, cN, color=C3, lw=1.2, marker="o", ms=5, mec="white", mew=0.8)
ax2.set_xlabel(r"$N$"); ax2.set_ylabel(r"$(2\sqrt{2}-\mathcal{B}_{\rm CHSH})/\sigma^2$")
ax2.set_xticks(Ns); ax2.set_ylim(0, 23)
ax2.text(0.05, 0.04, r"(b) $\sigma=10^{-3}$", transform=ax2.transAxes, fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig_gaussian_seeds.pdf"); plt.close(fig)
print("ok")
