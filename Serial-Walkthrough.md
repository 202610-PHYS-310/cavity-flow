# Cavity flow on one process, step by step

This walks through `cavity_serial.py` and the arithmetic it calls in `cavity_core.py`. It is Barba's Step 11, reorganised so the same arithmetic can also run in parallel. Read this before `MPI-Walkthrough.md`: everything the parallel version does is a change to something described here.

## Step 1: the problem

A 2 by 2 box is full of fluid. The bottom and both sides are fixed walls; the lid slides to the right at speed 1. The lid drags the fluid next to it, and the fluid settles into a large vortex filling the box.

We solve for three fields on the grid: the velocity components u (in x) and v (in y), and the pressure p. The equations are the incompressible Navier-Stokes equations in two dimensions:

    du/dt + u du/dx + v du/dy = -(1/rho) dp/dx + nu (d2u/dx2 + d2u/dy2)
    dv/dt + u dv/dx + v dv/dy = -(1/rho) dp/dy + nu (d2v/dx2 + d2v/dy2)

and the condition that the fluid is incompressible, du/dx + dv/dy = 0.

There is no equation that says "dp/dt = ...". Pressure has no evolution of its own: at each instant it takes whatever values keep the flow incompressible. Taking the divergence of the two momentum equations and demanding that the divergence stays zero gives an equation for p instead, a Poisson equation:

    d2p/dx2 + d2p/dy2 = b

where b depends only on the velocities (step 6). So every time step has two jobs: find the pressure that goes with the current velocities, then use it to move the velocities forward.

## Step 2: the grid and the arrays

```python
n = int(sys.argv[1]) if len(sys.argv) > 1 else 41
dx, dt = cc.grid(n)
u, v, p, b = (np.zeros((n, n)) for _ in range(4))
```

The box is covered by n by n points, with spacing dx = 2 / (n - 1) in both directions. The code uses dx for dy everywhere; that is only right because the grid is square.

The arrays are indexed `[y, x]`, as in Barba's notebook: `u[j, i]` is the point in row j (height y = j dx) and column i (x = i dx). So row 0 is the bottom wall, the last row is the lid, and column 0 is the left wall. It is easy to read `u[j, i]` as "x first" and get the walls swapped; if a plot comes out rotated, this is why.

## Step 3: the time step, and why it shrinks

```python
def grid(n):
    dx = L / (n - 1)
    dt = min(0.001, 0.25 * dx * dx / NU)
    return dx, dt
```

The method is explicit: each new value is computed from the old ones, with no equation to solve for it. Explicit diffusion is stable only if dt is below about dx^2 / (4 nu). On Barba's 41 by 41 grid that limit is 0.00625, six times her 0.001, so it never matters. But it shrinks with the square of dx: at 1025 by 1025 it is about 9.5e-6, and a run at dt = 0.001 would blow up within a few steps. `grid()` uses Barba's 0.001 when it is safe and the limit when it is not.

The cost of this is worth noticing. Halving dx makes four times as many points and needs four times as many time steps to reach the same physical time: sixteen times the work.

## Step 4: the boundary conditions on the velocity

```python
u[-1, :] = cc.LID
```

That is the only line. The fluid sticks to the walls (no slip), so u = v = 0 on the bottom and sides, which is what `np.zeros` already gave. On the lid u = 1 and v = 0.

Nothing in the time loop touches these values again. Every update below covers only the interior, rows 1 to n - 2 and columns 1 to n - 2, so the edge values set here never change. Barba resets the walls after every step instead; the result is the same. (The two top corners belong to the lid row, so they move at speed 1 too, as in Barba's code.)

## Step 5: neighbours as shifted slices

Every function in `cavity_core.py` starts the same way:

```python
r, up, dn = slice(a, z), slice(a + 1, z + 1), slice(a - 1, z - 1)
```

This is the one idea to get comfortable with before reading the rest. To compute something at every interior point at once, NumPy needs whole arrays, not loops over i and j. A neighbour's value is the same array, shifted by one:

| Expression | Means, at each point (j, i) |
|---|---|
| `f[r, 1:-1]` | the point itself |
| `f[r, 2:]` | the neighbour to the right, (j, i+1) |
| `f[r, :-2]` | the neighbour to the left, (j, i-1) |
| `f[up, 1:-1]` | the neighbour above, (j+1, i) |
| `f[dn, 1:-1]` | the neighbour below, (j-1, i) |

All five have the same shape, so they line up point for point. A centred difference for du/dx is then `(u[r, 2:] - u[r, :-2]) / (2 * dx)`: right minus left, over 2 dx, everywhere in one line.

The rows run from `a` to `z - 1`. The serial code always passes `a = 1, z = n - 1`, the whole interior. They are arguments rather than fixed numbers so that the MPI code can pass a single strip; see step 5 of `MPI-Walkthrough.md`.

## Step 6: the source term

```python
cc.source(b, u, v, dx, dt, 1, n - 1)
```

`source()` computes the right-hand side of the pressure equation from the current velocities:

    b = rho ( (du/dx + dv/dy) / dt - (du/dx)^2 - 2 (du/dy)(dv/dx) - (dv/dy)^2 )

with every derivative a centred difference. b is computed once per time step, because the velocities do not change while the pressure is being found.

**Think of b as a charge density.** The equation d2p/dx2 + d2p/dy2 = b has the form of Poisson's equation in electrostatics, with p playing the potential and b the charge, apart from a sign. Where b is negative, p has a local maximum; where b is positive, p has a local minimum. So b marks the places where the flow needs a pressure peak or a pressure hole, and solving the Poisson equation spreads that need over the whole box. b has two parts, and they do different jobs.

**The physics part.** The velocity-gradient terms describe what the flow is doing. Split the velocity gradient into its symmetric part, the strain (the fluid being stretched or squashed), and its antisymmetric part, the rotation. A line of algebra shows that

    (du/dx)^2 + 2 (du/dy)(dv/dx) + (dv/dy)^2 = S:S - w^2 / 2

where S:S is the sum of the squares of the strain-rate components and w = dv/dx - du/dy is the vorticity. So the physics part of b is rho (w^2/2 - S:S), and for a flow that is exactly incompressible it is the whole of b:

    d2p/dx2 + d2p/dy2 = rho ( w^2 / 2 - S:S )

- **Where rotation wins**, in the core of a vortex, the right side is positive and the pressure has a minimum. That is the centripetal force: fluid going round in a circle needs a net push toward the centre, and a pressure low supplies it. It is why the eye of a hurricane is a low.
- **Where strain wins**, where the flow runs into a wall and has to turn, the right side is negative and the pressure has a maximum. The fluid arriving at the wall must be slowed and turned aside, and a pressure high does it.

In a pure shear, such as the thin layer just under the lid where u changes quickly with y, the two cancel exactly. That is why the rotation and strain terms are each several times bigger than their sum.

**The divergence part.** The other term, rho (du/dx + dv/dy) / dt, is feedback. If the velocities were exactly divergence-free it would vanish. Suppose instead that fluid is leaving some point, so the divergence there is positive. Then b is pushed up, which makes a pressure low at that point, and in the velocity update the pressure gradient accelerates fluid back toward it, cancelling the outflow. Where fluid is piling up, the opposite happens. The 1/dt asks for the error to be undone in a single step. The physics part is a prediction of the pressure the flow needs; the divergence part measures what the last step actually did and corrects for whatever the prediction missed.

**Look at b.** `plot_source.py` draws b, its two parts, and the pressure each part makes on its own:

    python plot_source.py cavity-serial-41.npz

The Poisson equation is linear, so the pressure from the divergence part plus the pressure from the physics part is the run's pressure. For Barba's run the picture is not the one the physics alone suggests:

| | rms of the source | pressure it makes |
|---|---|---|
| divergence part | about 100 | -2.8 to +3.0 |
| physics part | about 0.9 | -0.04 to +0.05 |

The divergence part is most of b, and it is concentrated in the top few rows, in bands that alternate in sign along the lid. Both corner pressures, the high at the top right and the low at the top left, come from it. The largest divergence in the box is about 2.3 per second, and the 1/dt = 1000 turns that into a source of thousands.

The reason is the lid. At its two ends the velocity jumps from 1 to 0 between neighbouring points, and the real solution is singular there: the velocity gradients, and the pressure, have no finite value at those corners. No grid resolves that, and the divergence is largest in the cells beside them. Refining the grid makes it worse, not better (the largest divergence is about 0.4 at 21 by 21, 2.3 at 41 by 41 and 9 at 81 by 81). Nor is it a sign that the pressure solve has not converged: 500 or 5000 sweeps per step instead of 50 give the same answer.

Away from the corners the physics part does its job. On its own it makes a low in the middle of the vortex, about -0.04, which is most of the pressure there, and highs at both top corners, where the lid drives fluid into one wall and pulls it away from the other, and in both places the flow has to turn. The divergence part overrules it only at the top left, where it makes a low about eighty times bigger.

**Why pressure needs no time step.** Poisson's equation has no propagation speed: change b anywhere and p changes everywhere at once. That is what incompressible means. The fluid behaves as if sound travelled infinitely fast, so pressure is not something that evolves; it is whatever it must be, at each instant, to keep the flow divergence-free, as nearly as the grid allows.

## Step 7: the pressure, by Jacobi iteration

```python
for it in range(cc.NIT):
    pn = p.copy()
    cc.jacobi(p, pn, b, dx, 1, n - 1)
    ...boundary conditions, step 8...
```

Write the Poisson equation with centred second differences and solve it for the middle point:

    (p_right + p_left + p_up + p_down - 4 p) / dx^2 = b
    p = (p_right + p_left + p_up + p_down - dx^2 b) / 4

That is what `jacobi()` does at every interior point. Doing it once does not solve the equation, because the neighbours on the right-hand side are themselves wrong. Doing it repeatedly converges toward the solution: each sweep makes every point agree a little better with its neighbours.

Two details:

- `pn = p.copy()` keeps the values from before the sweep, and every point reads only those. That is what makes it Jacobi. Using new values as soon as they are computed would be Gauss-Seidel, which converges faster but depends on the order the points are visited in. Jacobi does not, which is what later lets the MPI version match this one exactly.
- The number of sweeps is fixed at `NIT = 50`, as in Barba. Nothing checks whether 50 is enough. The README suggests checking.

The previous time step's pressure is where the iteration starts. The flow changes little in one step, so that is already a good guess, and 50 sweeps only have to correct it.

## Step 8: the boundary conditions on the pressure

```python
p[:, -1] = p[:, -2]   # dp/dx = 0 at x = 2
p[0, :] = p[1, :]     # dp/dy = 0 at y = 0
p[:, 0] = p[:, 1]     # dp/dx = 0 at x = 0
p[-1, :] = 0          # p = 0 at y = 2
```

These run after every sweep, because each sweep changes the interior values next to the walls. On the three fixed walls the pressure gradient across the wall is zero, which the code makes true by copying the next row or column in. On the lid the pressure is set to 0. Only pressure differences matter in the momentum equations, so this just fixes the level; without it p would be determined only up to a constant.

The four lines overlap at the corners, so their order decides the corner values. This is Barba's order. Changing it changes the corners slightly, which barely affects the flow but does matter for the exact comparison with the MPI code.

## Step 9: the velocity update

```python
un, vn = u.copy(), v.copy()
cc.velocity(u, v, un, vn, p, dx, dt, 1, n - 1)
```

`velocity()` steps each momentum equation forward by dt, with every term from step 1 written as a difference:

| Term | Written as |
|---|---|
| du/dt | (u_new - u_old) / dt, forward in time |
| u du/dx, v du/dy (convection) | a backward difference: the point minus the one to its left, or below |
| dp/dx | centred: right minus left, over 2 dx |
| d2u/dx2 + d2u/dy2 (diffusion) | the five-point Laplacian: the four neighbours minus 4 times the point, over dx^2 |

The backward differences for convection are Barba's choice. They are "upwind", taking information from the side the flow comes from, wherever u and v are positive. In the lower part of the box the vortex carries the fluid to the left, so there they are downwind instead. The scheme stays stable here because viscosity is large enough to damp what that would excite. At a smaller viscosity it may not.

The copies `un` and `vn` play the same part as `pn`: every point is updated from the old values only.

## Step 10: timing, and what to look at

```python
t0 = time.perf_counter()
...source and pressure...
t1 = time.perf_counter()
...velocity...
```

The script times the two halves of the step separately and reports the split. On every grid size tried, about 90 % of the time is the pressure. Count the work per point to see why: the velocity update is one pass of perhaps 40 operations, and the pressure is 50 passes, each with a copy, a five-point average and four boundary assignments.

At the end it saves u, v and p to `cavity-serial-<n>.npz`. Draw them with

    python plot_cavity.py cavity-serial-41.npz

and compare with the last figure in Barba's Step 11 notebook. You should see the vortex centred in the upper half of the box, near x = 1, a high pressure in the top right corner, where the lid drives fluid into the wall, and a low pressure in the top left, where it pulls fluid away. Step 6 shows that both corner pressures come mostly from the divergence part of b; `plot_source.py` draws where each part of the pressure comes from.

## Things to check yourself

- Run 41 by 41 for 700 steps, then 1400. Has the flow stopped changing, or is Barba's 700-step picture still on its way somewhere?
- Print the largest value of |du/dx + dv/dy| at the end of a run, and the largest value more than a few cells from the lid. How close to incompressible is the answer, and where is it worst?
- Set `NU = 0.01` and run again. Does it stay stable? If not, which term is to blame?
