"""Draw the source term b of the pressure equation, its two parts, and the
pressure each part makes on its own. Walkthrough Step 6 explains them.

    python plot_source.py cavity-serial-41.npz      # writes cavity-serial-41-source.png

    b = rho (du/dx + dv/dy) / dt  +  rho (w^2/2 - S:S)
          divergence part              physics part

Top row: b and its two parts. Red is where b > 0, which asks for a pressure
low; blue is where b < 0, which asks for a high.

Bottom row: the run's pressure, then the pressure you get by solving the
Poisson equation with only one part of b as the source. The equation is
linear, so the two add up to the run's pressure.

Needs matplotlib, like plot_cavity.py. The two extra Poisson solves use the
code's own Jacobi sweep, run to convergence, so they take a few seconds at
41 by 41 and much longer on fine grids.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import cavity_core as cc

f = np.load(sys.argv[1])
u, v, p = f["u"], f["v"], f["p"]
n = u.shape[0]
dx, dt = cc.grid(n)

# The same centred differences as cc.source(), on the interior.
dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
dvdy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)
dudy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
dvdx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)

div = cc.RHO * (dudx + dvdy) / dt
rot = cc.RHO * 0.5 * (dvdx - dudy) ** 2                              # rotation
strain = -cc.RHO * (dudx**2 + dvdy**2 + 0.5 * (dudy + dvdx) ** 2)    # strain
phys = rot + strain

b = np.zeros((n, n))
cc.source(b, u, v, dx, dt, 1, n - 1)
assert np.allclose(b[1:-1, 1:-1], div + phys), "the split does not add up to b"


def pressure_from(src, sweeps=12 * n * n):
    """Solve the pressure equation with src as the source, with the same
    sweep and boundary conditions as cavity_serial.py, run to convergence."""
    bb = np.zeros((n, n))
    bb[1:-1, 1:-1] = src
    q = np.zeros((n, n))
    for _ in range(sweeps):
        qn = q.copy()
        cc.jacobi(q, qn, bb, dx, 1, n - 1)
        q[:, -1] = q[:, -2]
        q[0, :] = q[1, :]
        q[:, 0] = q[:, 1]
        q[-1, :] = 0
    return q


p_div = pressure_from(div)
p_phys = pressure_from(phys)

X, Y = np.meshgrid(np.linspace(0, cc.L, n), np.linspace(0, cc.L, n))
Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]

# The two top corners, where the lid meets the walls, are singular: b is huge
# in the cells beside them. Scale the colours to the rest of the box and let
# the corners saturate. The physics part gets its own scale, or it would
# vanish beside the divergence part.
lim_b = np.percentile(np.abs(b), 98)
lim_phys = np.percentile(np.abs(phys), 98)
lim_p = np.abs(p).max()
lim_pp = np.abs(p_phys).max()

top = [(Xi, Yi, b[1:-1, 1:-1], lim_b, "b, the whole source"),
       (Xi, Yi, div, lim_b, "divergence part:  rho (du/dx + dv/dy) / dt"),
       (Xi, Yi, phys, lim_phys, "physics part:  rho (w^2/2 - S:S)   [own scale]")]
bottom = [(X, Y, p, lim_p, "p, from the run"),
          (X, Y, p_div, lim_p, "p from the divergence part alone"),
          (X, Y, p_phys, lim_pp, "p from the physics part alone   [own scale]")]

fig, axs = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)
for row, panels, cmap, label in [
        (0, top, "RdBu_r", "red: asks for a pressure low; blue: a high"),
        (1, bottom, "PuOr_r", "pressure")]:
    for ax, (x, y, field, lim, title) in zip(axs[row], panels):
        m = ax.pcolormesh(x, y, field, cmap=cmap, vmin=-lim, vmax=lim,
                          shading="nearest")
        ax.contour(X, Y, p, colors="k", linewidths=0.5, levels=12)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(m, ax=ax, shrink=0.8, orientation="horizontal", label=label)
fig.suptitle("Black lines: contours of the run's pressure, the same in every panel")
out = sys.argv[1].replace(".npz", "-source.png")
fig.savefig(out, dpi=110)
print("wrote", out)

rms = lambda a: np.sqrt(np.mean(a**2))
print(f"rms of b's parts:   divergence {rms(div):8.2f}   physics {rms(phys):6.2f}"
      f"   (rotation {rms(rot):.2f}, strain {rms(strain):.2f}: they mostly cancel)")
print(f"largest |du/dx + dv/dy|: {np.abs(div * dt / cc.RHO).max():.3f} per second")
print(f"pressure range: run {p.min():.2f} to {p.max():.2f},"
      f"  from divergence {p_div.min():.2f} to {p_div.max():.2f},"
      f"  from physics {p_phys.min():.3f} to {p_phys.max():.3f}")
