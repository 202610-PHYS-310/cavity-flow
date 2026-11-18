"""Flow around a corner, on one process, run until it stops changing.

    python corner_serial.py              # 41 x 41, up to 20000 steps
    python corner_serial.py 81 60000     # finer grid, more steps allowed

Every 500 steps it prints the time, the largest change in u or v per unit time
(it falls towards zero as the flow becomes steady), and the volume flow in and
out. It stops when the change is below 1e-3, and saves the fields to
corner-serial-<n>.npz (plot_corner.py draws them).
"""
import sys

import numpy as np

import cavity_core as cc
import corner_core as k

n = int(sys.argv[1]) if len(sys.argv) > 1 else 41
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
dx, dt, m = k.grid(n)
u_in = k.inflow(n, dx, m)

u, v, p, b = (np.zeros((n, n)) for _ in range(4))
k.velocity_bc(u, v, n, dx, m, u_in)
print(f"corner  n {n}  dx {dx:.4f}  dt {dt:.2e}  Re {k.U * k.W / k.NU:.0f}")

for step in range(1, nt + 1):
    cc.source(b, u, v, dx, dt, 1, n - 1)
    for it in range(k.NIT):
        pn = p.copy()
        cc.jacobi(p, pn, b, dx, 1, n - 1)
        k.pressure_bc(p, m)
    un, vn = u.copy(), v.copy()
    k.velocity(u, v, un, vn, p, dx, dt)
    k.velocity_bc(u, v, n, dx, m, u_in)
    change = max(np.abs(u - un).max(), np.abs(v - vn).max()) / dt
    if not np.isfinite(change):
        sys.exit(f"step {step}: the solution blew up")
    if step % 500 == 0 or change < 1e-3:
        q_in, q_out = k.fluxes(u, v, dx, m)
        print(f"t {step * dt:7.2f}  change {change:9.2e}  flow in {q_in:.4f}  out {q_out:.4f}")
    if change < 1e-3:
        print(f"steady after {step} steps")
        break
else:
    print(f"not steady after {nt} steps; run longer")

np.savez(f"corner-serial-{n}.npz", u=u, v=v, p=p, m=m)
