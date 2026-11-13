# From serial to MPI: cavity flow step by step

This walks through `cavity_mpi.py`, starting from `cavity_serial.py` and changing one thing at a time. Read the serial code first; it is short, and `Serial-Walkthrough.md` explains it. If you have read `jacobi_mpi.py`, steps 1 to 6 will look familiar. Steps 7 to 9 are what the Navier-Stokes solver adds.

## Step 1: find what each point reads

Before splitting anything, ask what one point needs. Every update in `cavity_core.py` has the same shape. To compute a new value at row j, column i, it reads the old values at rows j-1, j and j+1, columns i-1, i and i+1. Nothing farther away.

That is what makes domain decomposition possible. If you cut the grid into pieces, a point needs only its own piece plus one row or column of its neighbour's. The Laplace equation in `jacobi_mpi.py` had the same property.

## Step 2: choose how to cut

The arrays are indexed `[y, x]`, as in Barba's notebook: `u[j, i]` is row j (height y) and column i. NumPy stores a 2-D array one row after another, so a whole row is one contiguous block of memory and can be sent as a message with no copying. A column is spread across the array.

So we cut the box into horizontal strips of whole rows, one strip per rank. `jacobi_mpi.py` cut its grid into strips too, for the same reason; there the first index was x, so its strips were vertical.

Strips are the simplest cut, not the only one. Blocks send less data per rank once there are many ranks. See "Things to try" in the README.

## Step 3: share out the rows

```python
counts = [n // size + (k < n % size) for k in range(size)]
g0, m = sum(counts[:rank]), counts[rank]
```

`counts` gives every rank n // size rows, and the first n % size ranks one extra. With n = 41 on 4 ranks that is 11, 10, 10, 10. Each rank owns m rows, starting at global row `g0`.

Unlike `jacobi_mpi.py`, the wall rows are shared out too. Rank 0 owns the bottom wall (global row 0) and the last rank owns the lid (global row n - 1). That keeps each boundary condition on exactly one rank.

The script refuses to run if any rank would get fewer than 2 rows. The reason comes up in step 8.

## Step 4: add ghost rows

```python
u, v, p, b = (np.zeros((m + 2, n)) for _ in range(4))
```

Each local array has m + 2 rows: the m the rank owns, in local rows 1 to m, plus one ghost row on each side. Local row 0 holds a copy of the last row of the strip below, and local row m + 1 a copy of the first row of the strip above. Local row k is global row `g0 + k - 1`.

A ghost row is never computed here. It is only ever filled by a message, and it is only correct right after one.

## Step 5: decide which rows to update

```python
a = 2 if first else 1
z = m if last else m + 1
```

In the serial code the update covers rows 1 to n - 2: everything except the bottom wall and the lid. Here each rank updates its own rows, local `a` to `z - 1`, except that rank 0 skips its bottom wall (local row 1) and the last rank skips its lid (local row m).

Every function in `cavity_core.py` takes this row range. The serial code calls them with rows 1 to n - 2, the MPI code with `a` to `z - 1`. That sharing is deliberate: the same lines of arithmetic run in both, which is what lets step 10 demand an exact match.

## Step 6: the halo exchange

```python
def halo(f):
    comm.Sendrecv(f[m], dest=above, recvbuf=f[0], source=below)
    comm.Sendrecv(f[1], dest=below, recvbuf=f[m + 1], source=above)
```

The first call sends every rank's top owned row up and fills the ghost row at the bottom from the rank below. The second does the reverse. After both, every ghost row matches its neighbour's owned row.

Two details:

- `Sendrecv` sends and receives as one operation. With a separate `Send` and then `Recv`, every rank could sit in its `Send` waiting for its neighbour to start receiving, and none ever would. `Sendrecv` cannot deadlock that way.
- Rank 0 has nothing below it and the last rank has nothing above. Their neighbour is `MPI.PROC_NULL`, and a message to or from `PROC_NULL` does nothing. That lets every rank run the same two lines with no `if`.

## Step 7: place the exchanges in the time step

This is the step that is new compared with `jacobi_mpi.py`. A time step has three stages, and each stage reads some fields at its neighbours' points. Each of those fields needs fresh ghost rows before that stage runs:

| Stage | Reads at neighbouring rows | Exchange before it |
|---|---|---|
| Pressure source `b` | u, v | `halo(u)`, `halo(v)` |
| Each of the 50 Jacobi sweeps | p | `halo(p)`, every sweep |
| Velocity update | u, v, and the new p | `halo(p)` once more |

u and v do not change between the source and the velocity update, so one exchange of each serves both. p changes on every sweep, so its ghost rows go stale every sweep.

That is 53 exchanges per time step: 2 for the velocities, 50 inside the pressure loop and 1 after it. The wave equation needed one. This count is why small grids run no faster in parallel (see the README), and why the 50 fixed sweeps are worth questioning.

## Step 8: boundary conditions on a strip

```python
p[1:m + 1, -1] = p[1:m + 1, -2]   # dp/dx = 0 at x = 2
if first:
    p[1, :] = p[2, :]             # dp/dy = 0 at y = 0
p[1:m + 1, 0] = p[1:m + 1, 1]     # dp/dx = 0 at x = 0
if last:
    p[m, :] = 0                   # p = 0 at y = 2
```

The side walls run through every strip, so every rank applies those two lines to its own rows. The bottom wall belongs to rank 0 only and the lid to the last rank only.

Keep the order the serial code uses. The four lines overlap at the corners, and a different order gives different corner values, which spoils the exact match.

This is also why every rank needs at least 2 rows. The bottom condition copies global row 1 into global row 0, and it needs the value of row 1 from this sweep. If rank 0 owned only row 0, row 1 would be a ghost row holding the value from before the sweep.

The velocities need no boundary code in the loop at all. Only the rows from `a` to `z - 1` are updated, so the walls and the lid keep the values they were given at the start. The only setup is that the last rank sets its lid row to speed 1.

## Step 9: time it and gather the answer

```python
comm.Barrier()
t_start = time.perf_counter()
...
elapsed = comm.reduce(time.perf_counter() - t_start, op=MPI.MAX, root=0)
```

The barrier starts every rank's clock together. The run is only as fast as its slowest rank, so the time reported is the maximum over ranks, not rank 0's own time. `halo()` adds up the time spent in exchanges, and the script reports it for the slowest rank too.

At the end each rank sends its owned rows, without the ghosts, to rank 0. Rank 0 stacks them in order with `np.concatenate` and saves the result.

## Step 10: check it against the serial code

```
python compare.py 41 4
```

The answer should be identical to the serial run, a difference of exactly zero. Both codes run the same arithmetic on each point, and Jacobi reads only old values, so the order the points are visited in cannot matter. Any difference at all means a bug.

Close is not good enough. Here is what three bugs, each one missing exchange, do to the 41 x 41 run on 4 ranks:

| Bug | Largest difference from serial |
|---|---|
| No exchange of p after the last sweep | about 1e-7 |
| p exchanged once per time step instead of before every sweep | the run blows up (overflow, then NaN) |
| No exchange of u and v | about 1 in p, 0.15 in u and v |

The first bug is the dangerous one. A difference of 1e-7 is invisible in a plot, and the flow looks exactly like Barba's. Only an exact comparison catches it.
