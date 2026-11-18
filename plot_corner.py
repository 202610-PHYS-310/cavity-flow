"""Draw a saved corner run: pressure, velocity arrows and streamlines, with the
solid block shaded. Needs matplotlib, so run it on your own Python.

    python plot_corner.py corner-serial-41.npz      # writes corner-serial-41.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import corner_core as k

f = np.load(sys.argv[1])
u, v, p, m = f["u"], f["v"], f["p"], int(f["m"])
n = u.shape[0]
X, Y = np.meshgrid(np.linspace(0, k.L, n), np.linspace(0, k.L, n))
solid = (X < k.W) & (Y > k.W)
pm = np.ma.masked_where(solid, p)
um, vm = np.ma.masked_where(solid, u), np.ma.masked_where(solid, v)
s = max(1, n // 20)

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax in axs:
    c = ax.contourf(X, Y, pm, levels=20, alpha=0.5, cmap="viridis")
    ax.contour(X, Y, pm, levels=20, colors="k", linewidths=0.4)
    ax.fill([0, k.W, k.W, 0], [k.W, k.W, k.L, k.L], color="0.6")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0, k.L)
    ax.set_ylim(0, k.L)
fig.colorbar(c, ax=axs, label="pressure", shrink=0.8)
axs[0].quiver(X[::s, ::s], Y[::s, ::s], um[::s, ::s], vm[::s, ::s])
axs[0].set_title("velocity")
# Start streamlines across the inflow so each one carries the same share of
# the flow: equal spacing in the cumulative flow, not in y.
y = np.linspace(0, k.W, 400)
q = np.cumsum(6 * y * (k.W - y))
starts = np.interp(np.linspace(0.05, 0.95, 12) * q[-1], q, y)
axs[1].streamplot(X, Y, np.ma.filled(um, 0), np.ma.filled(vm, 0), color="k",
                  start_points=np.column_stack([np.full_like(starts, 0.01), starts]),
                  density=3, integration_direction="forward")
axs[1].streamplot(X, Y, np.ma.filled(um, 0), np.ma.filled(vm, 0), color="tab:red",
                  density=1.2, linewidth=0.6, arrowsize=0.6)
axs[1].set_title("streamlines (black: from the inflow)")
out = sys.argv[1].replace(".npz", ".png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
