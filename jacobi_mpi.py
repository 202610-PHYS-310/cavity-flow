"""Jacobi relaxation for Laplace's equation, split across MPI ranks.

Same problem as the week-1 demo: the unit square, u = sin(pi x) on the top
edge and 0 on the other three, exact solution
u = sinh(pi y) sin(pi x) / sinh(pi).

The square is cut into vertical strips, one per rank. Each sweep needs a
halo exchange with the neighbouring strips, like the wave equation, plus one
new thing: every rank must agree on when to stop, which takes a global
reduction (allreduce) every sweep.

    mpirun -n 4 python jacobi_mpi.py
"""
import sys
import time

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
TOL = 1e-6
MAX_SWEEPS = 200_000
h = 1.0 / N


def sweep(u):
    """One Jacobi sweep of the interior. Returns the largest change."""
    new = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])
    change = np.max(np.abs(new - u[1:-1, 1:-1]))
    u[1:-1, 1:-1] = new
    return change


# Split the interior columns i = 1 .. N-1 as evenly as possible.
counts = [(N - 1) // size + (k < (N - 1) % size) for k in range(size)]
i0, n = 1 + sum(counts[:rank]), counts[rank]

# Local array holds columns i0-1 .. i0+n: n owned columns and two ghosts.
# The first index is x, so each column u[i] is one contiguous row in memory.
x = (i0 - 1 + np.arange(n + 2)) * h
u = np.zeros((n + 2, N + 1))
u[:, N] = np.sin(np.pi * x)
left = rank - 1 if rank > 0 else MPI.PROC_NULL
right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

comm.Barrier()
t0 = time.perf_counter()

for k in range(1, MAX_SWEEPS + 1):
    comm.Sendrecv(u[n], dest=right, recvbuf=u[0], source=left)
    comm.Sendrecv(u[1], dest=left, recvbuf=u[n + 1], source=right)
    change = sweep(u)
    # The one global step: stop only when the largest change ANYWHERE is small.
    if comm.allreduce(change, op=MPI.MAX) < TOL:
        break

elapsed = comm.reduce(time.perf_counter() - t0, op=MPI.MAX, root=0)
strips = comm.gather(u[1:-1], root=0)

if rank == 0:
    xs = np.arange(N + 1) * h
    full = np.zeros((N + 1, N + 1))
    full[:, N] = np.sin(np.pi * xs)
    full[1:-1] = np.concatenate(strips)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    err = np.max(np.abs(full - np.sinh(np.pi * Y) * np.sin(np.pi * X) / np.sinh(np.pi)))

    # The same sweeps on one array. Jacobi reads only the old values, so the
    # order points are updated in cannot matter: the answers must agree exactly.
    s = np.zeros((N + 1, N + 1))
    s[:, N] = np.sin(np.pi * xs)
    for ks in range(1, MAX_SWEEPS + 1):
        if sweep(s) < TOL:
            break

    print(f"ranks {size}  N {N}  sweeps {k} (serial {ks})")
    print(f"max error vs exact solution: {err:.2e}")
    print(f"identical to serial run:     {np.array_equal(full, s)}")
    print(f"wall time:                   {elapsed:.3f} s")
