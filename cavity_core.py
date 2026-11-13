"""The arithmetic of Barba's Step 11 (cavity flow), shared by the serial and MPI codes.

Arrays are indexed [row, column] = [y, x], as in Barba's notebook, so row 0 is
the bottom wall and the last row is the moving lid.

Every function works on a band of rows, rows a to b-1 of the array it is given,
and reads one row above and one below. The serial code passes the whole
interior. An MPI rank passes the rows it owns, with its ghost rows supplying
the neighbours. Both run exactly the same floating-point operations on each
point, which is why their answers can be compared for exact equality.
"""

RHO = 1.0   # density
NU = 0.1    # kinematic viscosity
LID = 1.0   # speed of the lid
L = 2.0     # the box is L by L
NIT = 50    # pressure iterations per time step, fixed, as in Barba


def grid(n):
    """Grid spacing and a stable time step for an n by n grid."""
    dx = L / (n - 1)
    # Barba's dt = 0.001 at n = 41. Finer grids need a smaller step to stay
    # stable: explicit diffusion needs dt below about dx^2 / (4 nu).
    dt = min(0.001, 0.25 * dx * dx / NU)
    return dx, dt


def source(b, u, v, dx, dt, a, z):
    """Right-hand side of the pressure equation, rows a..z-1."""
    r, up, dn = slice(a, z), slice(a + 1, z + 1), slice(a - 1, z - 1)
    dudx = (u[r, 2:] - u[r, :-2]) / (2 * dx)
    dvdy = (v[up, 1:-1] - v[dn, 1:-1]) / (2 * dx)
    dudy = (u[up, 1:-1] - u[dn, 1:-1]) / (2 * dx)
    dvdx = (v[r, 2:] - v[r, :-2]) / (2 * dx)
    b[r, 1:-1] = RHO * ((dudx + dvdy) / dt - dudx**2 - 2 * dudy * dvdx - dvdy**2)


def jacobi(p, pn, b, dx, a, z):
    """One Jacobi sweep of the pressure Poisson equation, rows a..z-1.
    Reads the old values pn, writes p."""
    r, up, dn = slice(a, z), slice(a + 1, z + 1), slice(a - 1, z - 1)
    p[r, 1:-1] = 0.25 * (pn[r, 2:] + pn[r, :-2] + pn[up, 1:-1] + pn[dn, 1:-1]
                         - dx * dx * b[r, 1:-1])


def velocity(u, v, un, vn, p, dx, dt, a, z):
    """One time step of the momentum equations, rows a..z-1.
    Reads the old velocities un, vn and the new pressure p."""
    r, up, dn = slice(a, z), slice(a + 1, z + 1), slice(a - 1, z - 1)
    c = (r, slice(1, -1))
    lap_u = un[r, 2:] + un[r, :-2] + un[up, 1:-1] + un[dn, 1:-1] - 4 * un[c]
    lap_v = vn[r, 2:] + vn[r, :-2] + vn[up, 1:-1] + vn[dn, 1:-1] - 4 * vn[c]
    # Backward differences for the convection terms (upwind only where the
    # velocity is positive), centred for the rest.
    u[c] = (un[c]
            - un[c] * dt / dx * (un[c] - un[r, :-2])
            - vn[c] * dt / dx * (un[c] - un[dn, 1:-1])
            - dt / (2 * RHO * dx) * (p[r, 2:] - p[r, :-2])
            + NU * dt / (dx * dx) * lap_u)
    v[c] = (vn[c]
            - un[c] * dt / dx * (vn[c] - vn[r, :-2])
            - vn[c] * dt / dx * (vn[c] - vn[dn, 1:-1])
            - dt / (2 * RHO * dx) * (p[up, 1:-1] - p[dn, 1:-1])
            + NU * dt / (dx * dx) * lap_v)
