"""Barba's Step 11: flow in a box with a moving lid, on one process.

    python cavity_serial.py              # 41 x 41, 700 steps: Barba's run
    python cavity_serial.py 1025 20      # a big grid, 20 steps, for timing

Prints how long a time step takes and how that splits between the pressure
solve and the velocity update, and saves the final fields to
cavity-serial-<n>.npz (plot_cavity.py draws them).
"""
import sys
import time

import numpy as np

import cavity_core as cc

n = int(sys.argv[1]) if len(sys.argv) > 1 else 41
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 700
dx, dt = cc.grid(n)

u, v, p, b = (np.zeros((n, n)) for _ in range(4))
u[-1, :] = cc.LID
t_pressure = t_velocity = 0.0

for step in range(nt):
    t0 = time.perf_counter()
    cc.source(b, u, v, dx, dt, 1, n - 1)
    for it in range(cc.NIT):
        pn = p.copy()
        cc.jacobi(p, pn, b, dx, 1, n - 1)
        p[:, -1] = p[:, -2]   # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]     # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]     # dp/dx = 0 at x = 0
        p[-1, :] = 0          # p = 0 at y = 2
    t1 = time.perf_counter()
    un, vn = u.copy(), v.copy()
    cc.velocity(u, v, un, vn, p, dx, dt, 1, n - 1)
    # The walls: no slip, and the lid moves at speed LID. Only the interior
    # was updated, so the edge values set at the start never change.
    t_pressure += t1 - t0
    t_velocity += time.perf_counter() - t1

total = t_pressure + t_velocity
print(f"serial  n {n}  steps {nt}  dt {dt:.2e}")
print(f"time per step:    {1e3 * total / nt:9.3f} ms")
print(f"  pressure solve: {100 * t_pressure / total:5.1f} %")
print(f"  velocity:       {100 * t_velocity / total:5.1f} %")
np.savez(f"cavity-serial-{n}.npz", u=u, v=v, p=p)
