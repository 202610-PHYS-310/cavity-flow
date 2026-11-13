"""Barba's Step 11 (cavity flow) split across MPI ranks.

    mpirun -n 4 python cavity_mpi.py              # 41 x 41, 700 steps
    mpirun -n 4 python cavity_mpi.py 1025 20      # a big grid, for timing

The box is cut into horizontal strips of rows, one strip per rank. Each rank
keeps one ghost row above its strip and one below, filled from its neighbours.

What one time step costs in messages:
  - one halo exchange of u and v, which the pressure source and the velocity
    update both read;
  - one halo exchange of p before each of the 50 pressure sweeps;
  - one more of p after the last sweep, for the velocity update.
That is 53 exchanges every time step, against one for the wave equation.

Saves the gathered fields to cavity-mpi-<n>-<ranks>.npz; compare.py checks
them against the serial run.
"""
import sys
import time

import numpy as np
from mpi4py import MPI

import cavity_core as cc

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

n = int(sys.argv[1]) if len(sys.argv) > 1 else 41
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 700
dx, dt = cc.grid(n)

# Share the n rows out as evenly as possible. Rank 0 holds the bottom wall,
# the last rank the lid.
counts = [n // size + (k < n % size) for k in range(size)]
if min(counts) < 2:
    sys.exit("each rank needs at least 2 rows: use fewer ranks or a bigger grid")
g0, m = sum(counts[:rank]), counts[rank]
first, last = rank == 0, rank == size - 1
below = rank - 1 if not first else MPI.PROC_NULL
above = rank + 1 if not last else MPI.PROC_NULL

# Local row k is global row g0 + k - 1; rows 0 and m + 1 are ghosts.
u, v, p, b = (np.zeros((m + 2, n)) for _ in range(4))
if last:
    u[m, :] = cc.LID
# The rows this rank updates: its own rows, minus the walls at the two ends.
a = 2 if first else 1
z = m if last else m + 1

t_comm = 0.0


def halo(f):
    """Fill the ghost rows of f from the neighbouring strips."""
    global t_comm
    t = time.perf_counter()
    comm.Sendrecv(f[m], dest=above, recvbuf=f[0], source=below)
    comm.Sendrecv(f[1], dest=below, recvbuf=f[m + 1], source=above)
    t_comm += time.perf_counter() - t


comm.Barrier()
t_start = time.perf_counter()

for step in range(nt):
    halo(u)
    halo(v)
    cc.source(b, u, v, dx, dt, a, z)
    for it in range(cc.NIT):
        halo(p)
        pn = p.copy()
        cc.jacobi(p, pn, b, dx, a, z)
        p[1:m + 1, -1] = p[1:m + 1, -2]   # dp/dx = 0 at x = 2
        if first:
            p[1, :] = p[2, :]             # dp/dy = 0 at y = 0
        p[1:m + 1, 0] = p[1:m + 1, 1]     # dp/dx = 0 at x = 0
        if last:
            p[m, :] = 0                   # p = 0 at y = 2
    halo(p)
    un, vn = u.copy(), v.copy()
    cc.velocity(u, v, un, vn, p, dx, dt, a, z)

elapsed = comm.reduce(time.perf_counter() - t_start, op=MPI.MAX, root=0)
comm_max = comm.reduce(t_comm, op=MPI.MAX, root=0)
parts = [comm.gather(f[1:m + 1], root=0) for f in (u, v, p)]

if rank == 0:
    u, v, p = (np.concatenate(s) for s in parts)
    print(f"ranks {size}  n {n}  steps {nt}  dt {dt:.2e}  rows per rank {min(counts)}-{max(counts)}")
    print(f"time per step:    {1e3 * elapsed / nt:9.3f} ms")
    print(f"  in halo exchange (slowest rank): {100 * comm_max / elapsed:5.1f} %")
    np.savez(f"cavity-mpi-{n}-{size}.npz", u=u, v=v, p=p)
