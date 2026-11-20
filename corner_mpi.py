"""Flow around a corner, split across MPI ranks, run until it stops changing.

    mpirun -n 4 python corner_mpi.py              # 41 x 41, up to 20000 steps
    mpirun -n 4 python corner_mpi.py 81 60000     # finer grid, more steps allowed

The box is cut into horizontal strips of rows, one per rank, exactly as
cavity_mpi.py cuts the cavity, and the interior arithmetic is the same:
cavity_core.source and cavity_core.jacobi, and corner_core.velocity, each
called on the rows a rank owns. What is new is the boundary conditions. In the
cavity they sat on the edges of the box, so each one belonged to the first
rank, the last rank, or every rank. Here the block, the inflow and the outflow
are inside the array, and a strip boundary can fall anywhere relative to them.

So every boundary condition in corner_core.py is rewritten below in terms of
GLOBAL rows, and each rank applies it only to the global rows it owns. Almost
every condition reads only the row it writes. The exceptions:

  - the bottom wall copies row 1, and the outflow copies row n-2: always on the
    same rank, because every rank owns at least 2 rows;
  - the bottom face of the block (row m) copies row m-1, and the block's corner
    averages in a value from row m-1. If a strip boundary falls between rows
    m-1 and m, row m-1 is on another rank, and it has just changed in this
    sweep. So that one row is sent, after every sweep, between those two ranks
    only. Every other rank skips the message (MPI.PROC_NULL).

Prints and stops exactly as corner_serial.py does, and saves the gathered
fields to corner-mpi-<n>-<ranks>.npz. Check it with

    python compare.py 41 4 corner
"""
import sys
import time

import numpy as np
from mpi4py import MPI

import cavity_core as cc
import corner_core as k

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

n = int(sys.argv[1]) if len(sys.argv) > 1 else 41
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
dx, dt, m = k.grid(n)
u_in = k.inflow(n, dx, m)

# Share the n rows out as evenly as possible, as in cavity_mpi.py.
counts = [n // size + (q < n % size) for q in range(size)]
if min(counts) < 2:
    sys.exit("each rank needs at least 2 rows: use fewer ranks or a bigger grid")
g0, cnt = sum(counts[:rank]), counts[rank]
first, last = rank == 0, rank == size - 1
below = rank - 1 if not first else MPI.PROC_NULL
above = rank + 1 if not last else MPI.PROC_NULL


def owner(g):
    """The rank that owns global row g."""
    return int(np.searchsorted(np.cumsum(counts), g, side="right"))


def mine(lo, hi):
    """Global rows lo..hi-1 that this rank owns, as (local slice, global slice).
    Local row j is global row g0 + j - 1; rows 0 and cnt + 1 are ghosts."""
    lo, hi = max(lo, g0), min(hi, g0 + cnt)
    if lo >= hi:
        return None
    return slice(lo - g0 + 1, hi - g0 + 1), slice(lo, hi)


def loc(g):
    """Local index of global row g (which this rank must own)."""
    return g - g0 + 1


def owns(g):
    return g0 <= g < g0 + cnt


# The rows this rank updates: its own rows, minus the walls at the two ends.
a = 2 if first else 1
z = cnt if last else cnt + 1
own = slice(1, cnt + 1)

# Row m-1 to row m, when they sit on different ranks (see the docstring).
A, B = owner(m - 1), owner(m)
face_dest = B if rank == A and A != B else MPI.PROC_NULL
face_src = A if rank == B and A != B else MPI.PROC_NULL


def velocity_bc(u, v):
    """corner_core.velocity_bc, statement by statement, on this rank's rows."""
    if owns(0):                                   # bottom wall
        u[loc(0), :] = v[loc(0), :] = 0
    u[own, -1] = v[own, -1] = 0                   # right wall
    if (r := mine(m, n)):                         # the block, its faces and corner
        u[r[0], :m + 1] = 0
        v[r[0], :m + 1] = 0
    if (r := mine(0, m + 1)):                     # inflow
        u[r[0], 0] = u_in[r[1]]
        v[r[0], 0] = 0
    if owns(n - 1):                               # outflow (row n-2 is ours too)
        u[loc(n - 1), m + 1:-1] = u[loc(n - 2), m + 1:-1]
        v[loc(n - 1), m + 1:-1] = v[loc(n - 2), m + 1:-1]


def pressure_bc(p):
    """corner_core.pressure_bc, statement by statement, on this rank's rows.
    The order matters (the outflow row overwrites the right wall), so it is
    kept exactly."""
    global t_comm
    if owns(0):                                   # bottom wall
        p[loc(0), :] = p[loc(1), :]
    p[own, -1] = p[own, -2]                       # right wall
    if (r := mine(0, m)):                         # inflow
        p[r[0], 0] = p[r[0], 1]
    # Row m-1 is final now. If row m lives on the next rank up, send it there,
    # into that rank's lower ghost row.
    t = time.perf_counter()
    comm.Sendrecv(p[loc(m - 1)] if face_dest != MPI.PROC_NULL else p[0], dest=face_dest,
                  recvbuf=p[0], source=face_src)
    t_comm += time.perf_counter() - t
    if owns(m):                                   # bottom face of the block
        p[loc(m), :m] = p[loc(m) - 1, :m]
    if (r := mine(m + 1, n)):                     # right face of the block
        p[r[0], m] = p[r[0], m + 1]
    if owns(m):                                   # the block's corner
        p[loc(m), m] = 0.5 * (p[loc(m) - 1, m] + p[loc(m), m + 1])
    if (r := mine(m + 1, n)):                     # inside the block: never used
        p[r[0], :m] = 0
    if owns(n - 1):                               # outflow
        p[loc(n - 1), m + 1:] = 0


t_comm = 0.0


def halo(f):
    """Fill the ghost rows of f from the neighbouring strips."""
    global t_comm
    t = time.perf_counter()
    comm.Sendrecv(f[cnt], dest=above, recvbuf=f[0], source=below)
    comm.Sendrecv(f[1], dest=below, recvbuf=f[cnt + 1], source=above)
    t_comm += time.perf_counter() - t


def fluxes(u, v):
    """corner_core.fluxes, on the root, from the pieces each rank owns."""
    col = comm.gather(u[own, 0], root=0)
    top = comm.bcast(v[loc(n - 1)] if last else None, root=size - 1)
    if rank == 0:
        full = np.zeros((n, n))
        full[:, 0] = np.concatenate(col)
        vv = np.zeros((n, n))
        vv[-1] = top
        return k.fluxes(full, vv, dx, m)
    return None, None


u, v, p, b = (np.zeros((cnt + 2, n)) for _ in range(4))
velocity_bc(u, v)
if rank == 0:
    print(f"corner  n {n}  dx {dx:.4f}  dt {dt:.2e}  Re {k.U * k.W / k.NU:.0f}  "
          f"ranks {size}  rows per rank {min(counts)}-{max(counts)}  "
          f"row m-1 and m on {'one rank' if A == B else f'ranks {A} and {B}'}")

comm.Barrier()
t_start = time.perf_counter()

for step in range(1, nt + 1):
    halo(u)
    halo(v)
    cc.source(b, u, v, dx, dt, a, z)
    for it in range(k.NIT):
        halo(p)
        pn = p.copy()
        cc.jacobi(p, pn, b, dx, a, z)
        pressure_bc(p)
    halo(p)
    un, vn = u.copy(), v.copy()
    k.velocity(u, v, un, vn, p, dx, dt, a, z)
    velocity_bc(u, v)
    mine_change = max(np.abs(u[own] - un[own]).max(), np.abs(v[own] - vn[own]).max()) / dt
    # Every rank must stop on the same step, so the test uses the largest
    # change anywhere, as in jacobi_mpi.py.
    change = comm.allreduce(mine_change, op=MPI.MAX)
    if not np.isfinite(change):
        if rank == 0:
            print(f"step {step}: the solution blew up")
        sys.exit(1)
    if step % 500 == 0 or change < 1e-3:
        q_in, q_out = fluxes(u, v)
        if rank == 0:
            print(f"t {step * dt:7.2f}  change {change:9.2e}  flow in {q_in:.4f}  out {q_out:.4f}")
    if change < 1e-3:
        if rank == 0:
            print(f"steady after {step} steps")
        break
else:
    if rank == 0:
        print(f"not steady after {nt} steps; run longer")

elapsed = comm.reduce(time.perf_counter() - t_start, op=MPI.MAX, root=0)
comm_max = comm.reduce(t_comm, op=MPI.MAX, root=0)
parts = [comm.gather(f[own], root=0) for f in (u, v, p)]

if rank == 0:
    u, v, p = (np.concatenate(s) for s in parts)
    print(f"time per step:    {1e3 * elapsed / step:9.3f} ms")
    print(f"  in messages (slowest rank): {100 * comm_max / elapsed:5.1f} %")
    np.savez(f"corner-mpi-{n}-{size}.npz", u=u, v=v, p=p, m=m)
