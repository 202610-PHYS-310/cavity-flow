"""Flow around a corner: an L-shaped channel, built on the cavity-flow arithmetic.

The box is 2 by 2, as in the cavity, but its top-left quarter is solid. Fluid
enters through the left side below the block, turns the corner, and leaves
through the top to the right of the block:

    y = 2  +---------+.........+    <- outflow (top, 1 < x < 2)
           |  solid  |         |
    y = 1  +---------+         |
    inflow |                   |
     -->   |                   |
    y = 0  +-------------------+
          x = 0     x = 1     x = 2

Both arms of the L are 1 wide. Arrays are indexed [row, column] = [y, x], as in
cavity_core.py. The pressure source and the Jacobi sweep are reused unchanged
from cavity_core.py; what is new is the geometry, the inflow and outflow
boundaries, and a convection term that is upwind in both directions.
"""
import numpy as np

import cavity_core as cc

L = cc.L      # the box is L by L
W = 1.0       # width of each arm of the L
U = 1.0       # mean inflow speed
NU = 0.1      # kinematic viscosity: Reynolds number U W / NU = 10
RHO = cc.RHO
NIT = cc.NIT


def grid(n):
    """Grid spacing, a stable time step, and the index m of x = W (and y = W)."""
    dx = L / (n - 1)
    m = round(W / dx)
    if abs(m * dx - W) > 1e-12:
        raise ValueError(f"n = {n} does not put a grid line at x = {W}; use n = 41, 81, 161, ...")
    # Explicit diffusion needs dt below about dx^2 / (4 nu); convection needs
    # dt below about dx / (largest speed), and the inflow peaks at 1.5 U.
    dt = min(0.001, 0.25 * dx * dx / NU, 0.5 * dx / (1.5 * U))
    return dx, dt, m


def inflow(n, dx, m):
    """The inflow profile: fully developed channel flow (a parabola) with mean
    speed U, across rows 0..m of column 0. Zero at both walls."""
    y = np.arange(m + 1) * dx
    return 6 * U * y * (W - y) / W**2


def velocity_bc(u, v, n, dx, m, u_in):
    """Walls, the block, the inflow and the outflow."""
    u[0, :] = v[0, :] = 0          # bottom wall
    u[:, -1] = v[:, -1] = 0        # right wall
    u[m:, :m + 1] = 0              # the block, its faces and corner
    v[m:, :m + 1] = 0
    u[:m + 1, 0] = u_in            # inflow: given profile, no y-velocity
    v[:m + 1, 0] = 0
    u[-1, m + 1:-1] = u[-2, m + 1:-1]   # outflow: nothing changes across it
    v[-1, m + 1:-1] = v[-2, m + 1:-1]


def pressure_bc(p, m):
    """dp/dn = 0 on every wall and at the inflow; p = 0 at the outflow."""
    p[0, :] = p[1, :]              # bottom wall
    p[:, -1] = p[:, -2]            # right wall
    p[:m, 0] = p[:m, 1]            # inflow
    p[m, :m] = p[m - 1, :m]        # bottom face of the block
    p[m + 1:, m] = p[m + 1:, m + 1]  # right face of the block
    p[m, m] = 0.5 * (p[m - 1, m] + p[m, m + 1])   # the block's corner
    p[m + 1:, :m] = 0              # inside the block: never used
    p[-1, m + 1:] = 0              # outflow


def velocity(u, v, un, vn, p, dx, dt):
    """One time step of the momentum equations on the whole interior.

    Same as cavity_core.velocity, except for the convection terms. Barba's code
    always takes the backward difference, which is upwind only where the
    velocity is positive. Here the flow turns, and the corner makes regions
    where it runs backwards, so each term takes the difference on the side the
    fluid is coming from."""
    c = (slice(1, -1), slice(1, -1))
    e, w = (slice(1, -1), slice(2, None)), (slice(1, -1), slice(None, -2))
    nn, s = (slice(2, None), slice(1, -1)), (slice(None, -2), slice(1, -1))

    def convect(f):
        dfdx = np.where(un[c] > 0, f[c] - f[w], f[e] - f[c]) / dx
        dfdy = np.where(vn[c] > 0, f[c] - f[s], f[nn] - f[c]) / dx
        return un[c] * dfdx + vn[c] * dfdy

    lap_u = un[e] + un[w] + un[nn] + un[s] - 4 * un[c]
    lap_v = vn[e] + vn[w] + vn[nn] + vn[s] - 4 * vn[c]
    u[c] = (un[c] - dt * convect(un)
            - dt / (2 * RHO * dx) * (p[e] - p[w])
            + NU * dt / (dx * dx) * lap_u)
    v[c] = (vn[c] - dt * convect(vn)
            - dt / (2 * RHO * dx) * (p[nn] - p[s])
            + NU * dt / (dx * dx) * lap_v)


def fluxes(u, v, dx, m):
    """Volume flow in through the left side and out through the top, per unit
    depth, by the trapezoid rule. In a steady incompressible flow they agree."""
    return np.trapezoid(u[:m + 1, 0], dx=dx), np.trapezoid(v[-1, m:], dx=dx)
