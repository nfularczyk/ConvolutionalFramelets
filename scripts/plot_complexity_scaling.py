"""Flop-model scaling with ell = N/4 and CR-matched reconstruction ranks (Fig. 6)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ConvFrameEX as ex
from complexity_comparison import matrix_flops, tensor_flops

out = ROOT / "outputs" / "fig_complexity_scaling.png"
Ns = np.array([32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256])
f1, f2 = [], []
for N in Ns:
    ell = int(N) // 4
    r, tens = ex.reconstruction_ranks(int(N), ell)
    f1.append(matrix_flops(int(N), ell, r).total)
    f2.append(tensor_flops(int(N), ell, tens).total)

f1 = np.asarray(f1)
f2 = np.asarray(f2)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
axes[0].loglog(Ns, f1, "o-", color="#4c78a8", label="1D framelets")
axes[0].loglog(Ns, f2, "s-", color="#f58518", label="2D framelets")
nref = Ns.astype(float)
axes[0].loglog(
    nref, f1[-1] / Ns[-1] ** 4 * nref**4, ":", color="#4c78a8", alpha=0.75, label=r"$\propto N^4$"
)
axes[0].loglog(
    nref, f2[-1] / Ns[-1] ** 5 * nref**5, ":", color="#f58518", alpha=0.75, label=r"$\propto N^5$"
)
axes[0].axvline(128, color="0.6", lw=0.8, ls="--")
axes[0].set_xlabel(r"image side $N$  ($\ell=N/4$)")
axes[0].set_ylabel("estimated flops")
axes[0].set_title("Total arithmetic vs image size")
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

ratio = f2 / f1
axes[1].plot(Ns, ratio, "D-", color="#54a24b", label="flop-model ratio")
# fit c*N through N=128
c = ratio[list(Ns).index(128)] / 128.0
axes[1].plot(Ns, c * Ns, "--", color="0.45", label=r"$\propto N$")
axes[1].axvline(128, color="0.6", lw=0.8, ls="--")
axes[1].set_xlabel(r"image side $N$  ($\ell=N/4$)")
axes[1].set_ylabel(r"$T_{\mathrm{2D}}/T_{\mathrm{1D}}$")
axes[1].set_title("Cost ratio grows like $O(N)$")
axes[1].legend(frameon=False)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
print("ratio N=128", float(ratio[list(Ns).index(128)]))
print("ratio N=256", float(ratio[-1]))
