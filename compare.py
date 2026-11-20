"""Check an MPI run against the serial one, point by point.

    python compare.py 41 4             # cavity-serial-41.npz against cavity-mpi-41-4.npz
    python compare.py 41 4 corner      # corner-serial-41.npz against corner-mpi-41-4.npz

Both codes run the same arithmetic on every point (cavity_core.py), and Jacobi
sweeps read only old values, so the order points are visited in cannot matter.
The fields should be identical, not merely close.
"""
import sys

import numpy as np

n, ranks = sys.argv[1], sys.argv[2]
case = sys.argv[3] if len(sys.argv) > 3 else "cavity"
s = np.load(f"{case}-serial-{n}.npz")
m = np.load(f"{case}-mpi-{n}-{ranks}.npz")
for k in ("u", "v", "p"):
    d = np.max(np.abs(s[k] - m[k]))
    print(f"{k}: largest difference {d:.1e}   identical: {np.array_equal(s[k], m[k])}")
