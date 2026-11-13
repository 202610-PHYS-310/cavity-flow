"""Draw a saved cavity run the way Barba does: pressure contours, velocity
arrows and streamlines. Needs matplotlib, which the MPI image does not have,
so run it on your own Python.

    python plot_cavity.py cavity-serial-41.npz      # writes cavity-serial-41.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import cavity_core as cc

f = np.load(sys.argv[1])
u, v, p = f["u"], f["v"], f["p"]
n = u.shape[0]
X, Y = np.meshgrid(np.linspace(0, cc.L, n), np.linspace(0, cc.L, n))
s = max(1, n // 20)   # about 20 arrows across, whatever the grid

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax in axs:
    c = ax.contourf(X, Y, p, alpha=0.5, cmap="viridis")
    ax.contour(X, Y, p, colors="k", linewidths=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
fig.colorbar(c, ax=axs, label="pressure", shrink=0.8)
axs[0].quiver(X[::s, ::s], Y[::s, ::s], u[::s, ::s], v[::s, ::s])
axs[0].set_title("velocity")
axs[1].streamplot(X, Y, u, v, color="k", density=1.2)
axs[1].set_title("streamlines")
out = sys.argv[1].replace(".npz", ".png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
