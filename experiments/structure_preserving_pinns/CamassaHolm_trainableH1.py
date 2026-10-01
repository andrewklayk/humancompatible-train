import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from time import time
from types import SimpleNamespace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--constrained', action='store_true')
parser.add_argument('--true_h0', action='store_true')
args = parser.parse_args()
CONSTRAINED = args.constrained
USE_TRUE_H0 = args.true_h0      # reference I^0 from the exact IC; else I(t=0) of the net

epochs = 3000

DTYPE = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

xMin = -np.pi
xMax = np.pi
tMax = 5.0

os.makedirs('./results/camassa', exist_ok=True)
save_fig = True

# ── Hard-encoded initial and boundary conditions (see PINN) ───────────────────
#
# Declared here rather than with the rest of the training setup because the
# periodic embedding changes the network's input dimension, which feeds the
# parameter count that the collocation grid is sized from.
HARD_IC = True     # u(.,0) = u_0 exactly, by construction
HARD_BC = True     # x-periodicity of u and all its derivatives, by construction
N_MODES = 8        # periodic feature modes K; network input dim becomes 2K+1
IC_TAU = 1.0       # phi(t) = 1 - exp(-t/tau)

D_IN = (2 * N_MODES + 1) if HARD_BC else 2

# ── Grid sizing from network architecture ─────────────────────────────────────

def Nx_from_arch(width, depth, fac=2.0, d_in=2, d_out=1):
    """
    Nx = Nt such that Nx * Nt ~= Ntheta / fac, rounded up so that Boole's rule
    tiles the x-grid exactly.

    `H_boole` needs (Nx - 1) % 4 == 0; on any other Nx the leftover nodes fall
    to a trapezoid tail. That tail spans a *fixed* number of intervals, so its
    absolute error is O(dx^3) and dominates -- the mixed rule converges at
    order 3, not Boole's 6. The plain sqrt gives Nx = 44 here, whose 3 leftover
    intervals put a 2.8e-4 relative error into H0 (4.2e-2 for the KdV script),
    i.e. larger than the drift tolerance the H constraint is tightened to.
    Rounding up to Nx = 45 makes the rule exact to machine precision.

    Returns (Nx, Nt, Ntheta, Ncoll_target). Note Nx * Nt then slightly exceeds
    Ncoll_target (2025 vs 1976 at the default), still far below Ntheta.
    """
    Ntheta = (d_in + 1) * width \
             + (depth - 1) * (width * width + width) \
             + d_out * (width + 1)
    Ncoll_target = int(Ntheta / fac)
    Nx = int(np.sqrt(Ncoll_target))
    Nx += (-(Nx - 1)) % 4        # next Nx with (Nx - 1) % 4 == 0
    return Nx, Nx, Ntheta, Ncoll_target


def h_from_NxNt(Nx, Nt, xMin, xMax, tMax):
    dx = (xMax - xMin) / (Nx - 1)
    dt = tMax / (Nt - 1)
    return dx, dt, max(dx, dt)


width, depth = 120, 4
Nx, Nt, Ntheta, Ncoll = Nx_from_arch(width, depth, fac=4.0, d_in=D_IN)
dx, dt, h = h_from_NxNt(Nx, Nt, xMin, xMax, tMax)

# ── Physics ───────────────────────────────────────────────────────────────────

def u_0(x):
    return 0.2 + 0.1 * torch.cos(2.0 * x)


def u_0_x(x):
    return -0.2 * torch.sin(2.0 * x)


# ── Conserved densities: the Camassa-Holm invariant hierarchy ─────────────────
#
# Written as a transport law for the momentum m = u - u_xx,
#
#     m_t + u m_x + 2 u_x m = 0,
#
# which expands to exactly the residual used below, CH conserves
#
#     I1 = int m dx    = int u dx                 (linear in u)
#     I2 = int u m dx  = int (u^2 + u_x^2) dx     (quadratic)
#     I3               = int (u^3 + u u_x^2) dx   (cubic)
#
# I3 is the Hamiltonian the original scripts constrained; `ch_density` keeps its
# old name and normalisation so H values stay comparable across runs. The
# lower-order members cost one quadrature each and are much better conditioned.

def ch_mass(u, u_x):
    """I1 density. int u_xx dx = 0 on a periodic domain, so int m dx = int u dx."""
    return u


def ch_h1(u, u_x):
    """I2 density (the H^1 energy), halved to match the usual normalisation."""
    return 0.5 * (u**2 + u_x**2)


def ch_density(u, u_x):
    """I3 density -- the Hamiltonian, in the original scripts' normalisation."""
    return u**3 + u * u_x**2


def quad_rect(f, dx):
    """Rectangular (1st-order) rule along the last dimension, per row."""
    return torch.sum(f, dim=-1) * dx

def H_rect(u, u_x, dx, density_fn=ch_density):
    """Rectangular (1st-order) rule along last dimension, integrating per row."""
    f = density_fn(u, u_x)
    return torch.sum(f, dim=-1) * dx


def quad_boole(f, dx):
    """
    Boole's rule (composite 5-point) along last dimension, trapezoid on tail.

    BUG FIX vs the TF original (changes results). TF slices block starts as
    `idx[0::4]` over arange(n1), which also picks up the *last* prefix node --
    already present in the block-end slice `idx[4::4]` -- so that node received
    weight 14 instead of 7. Composite Boole gives 14 only to the shared
    *interior* block boundaries; the two outer endpoints keep 7. `idx[0:-1:4]`
    restricts the slice to genuine block starts.

    The old value was too large by exactly `(2*dx/45) * 7 * f[n1-1]`, so H
    values here are no longer bit-comparable with earlier CH runs.
    """
    n = f.shape[-1]

    if n <= 1:
        return torch.sum(f, dim=-1) * dx

    n1 = n - ((n - 1) % 4)
    c = (2.0 * dx) / 45.0

    idx = torch.arange(n1, device=f.device)
    f0 = f[..., idx[0:-1:4]]   # block starts: 0, 4, ..., n1-5
    f1 = f[..., idx[1::4]]
    f2 = f[..., idx[2::4]]
    f3 = f[..., idx[3::4]]
    f4 = f[..., idx[4::4]]     # block ends:   4, 8, ..., n1-1

    # Per-block weights [7, 32, 12, 32, 7]; interior boundaries appear in both
    # f0 and f4 and so total 14, as they should.
    s = (7.0  * f0.sum(-1)
       + 32.0 * f1.sum(-1)
       + 12.0 * f2.sum(-1)
       + 32.0 * f3.sum(-1)
       + 7.0  * f4.sum(-1))
    boole = c * s

    if n1 == n:
        return boole

    tail = f[..., n1 - 1:]
    trap = torch.sum(0.5 * (tail[..., 1:] + tail[..., :-1]), dim=-1) * dx
    return boole + trap


def H_rect(u, u_x, dx, density_fn=ch_density):
    """Back-compatible wrapper: sample a density, then integrate it."""
    return quad_rect(density_fn(u, u_x), dx)


def H_boole(u, u_x, dx, density_fn=ch_density):
    """Back-compatible wrapper: sample a density, then integrate it."""
    return quad_boole(density_fn(u, u_x), dx)


# ── Model, with the initial and boundary conditions built in ──────────────────

class PeriodicFeatures(nn.Module):
    """
    x  ->  [sin(2 pi k (x - xMin)/L), cos(2 pi k (x - xMin)/L)]_{k=1..K}

    Every feature is L-periodic in x together with all of its derivatives, so
    any smooth function of them is exactly L-periodic. Feeding the network these
    instead of raw x makes the periodic BC hold by construction in u, u_x, u_xx,
    ... -- which is what a third-order equation actually needs (three
    conditions, where the soft version penalised two and the KdV script one).

    K does not band-limit the network: the hidden nonlinearities generate higher
    harmonics from these features (sin^2 = (1-cos2x)/2 and so on).
    """

    def __init__(self, n_modes, x_min, L):
        super().__init__()
        self.register_buffer(
            'k', torch.arange(1, n_modes + 1, dtype=DTYPE).reshape(1, -1))
        self.x_min = float(x_min)
        self.L = float(L)

    def forward(self, x):
        ang = (2.0 * np.pi / self.L) * self.k * (x - self.x_min)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class PINN(nn.Module):
    """
        u_theta(x, t) = u_0(x) + phi(t) * N_theta(gamma(x), t),
        phi(t) = 1 - exp(-t / tau),   phi(0) = 0

    so u_theta(x, 0) = u_0(x) exactly, and gamma (PeriodicFeatures) makes
    u_theta periodic in x exactly. Neither condition needs a penalty or a dual,
    which removes two constraint groups, two dual learning rates and the
    tolerance schedule for both from the tuning surface.

    This also removes the trivial-solution trap. u = 0 is an exact solution of
    the PDE *and* a critical point of the Hamiltonian (dI3/du = 3u^2 + u_x^2 -
    2(u u_x)_x vanishes there), so the soft formulation could park on it with a
    constraint gradient that dies to second order -- the reason the H constraint
    needed a grace period at all. Here it is simply not representable, since
    u(.,0) = u_0 is fixed and non-constant. What does remain reachable is
    N_theta = 0, i.e. the frozen field u(x,t) = u_0(x); that is exactly feasible
    for every invariant constraint, but it carries a large PDE residual, so the
    objective pushes away from it instead of toward it. Feasible-but-suboptimal
    is a much healthier failure mode than feasible-and-optimal.

    phi'(0) = 1/tau fixes the scale the network must produce to realise the
    initial time derivative: N_theta(x,0) = tau * u_t(x,0). tau ~ O(1) keeps
    that O(1). phi is bounded on [0,1), unlike phi(t) = t which would scale the
    network output by up to tMax.

    Full treatment in FORMULATION.md section 6: the lift, exactness, completeness
    (phi must have a SIMPLE zero at 0, or solutions with u_t(.,0) != 0 are
    excluded), the choice of tau and K, and the trap analysis.

    Set hard_ic / hard_bc False to recover the plain (x,t) -> u network that the
    soft formulation used.
    """

    def __init__(self, hidden_layers=depth, width=width,
                 hard_ic=None, hard_bc=None, n_modes=None, tau=None):
        super().__init__()
        self.hard_ic = HARD_IC if hard_ic is None else hard_ic
        self.hard_bc = HARD_BC if hard_bc is None else hard_bc
        self.tau = IC_TAU if tau is None else tau
        n_modes = N_MODES if n_modes is None else n_modes

        if self.hard_bc:
            self.feat = PeriodicFeatures(n_modes, xMin, xMax - xMin)
            d_in = 2 * n_modes + 1
        else:
            self.feat = None
            d_in = 2

        dims = [d_in] + [width] * hidden_layers + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        z = torch.cat([self.feat(x), t], dim=1) if self.hard_bc \
            else torch.cat([x, t], dim=1)
        out = self.net(z)
        if self.hard_ic:
            out = u_0(x) + (1.0 - torch.exp(-t / self.tau)) * out
        return out


# ── Loss helpers ──────────────────────────────────────────────────────────────

def lambda_grad_weight(epoch, start=epochs, lam_max=1.0, kappa=1e-3):
    """Ramps the H1 gradient penalty weight from 0 starting at `start` epochs."""
    return lam_max * (1.0 - np.exp(-kappa * max(float(epoch) - start, 0.0)))


def periodic_bc_loss(model, Nbc=2000):
    """Penalises mismatch in u and u_x between the left and right boundaries."""
    t = torch.rand(Nbc, 1, device=device, dtype=DTYPE) * tMax
    xL = torch.full((Nbc, 1), xMin, device=device, dtype=DTYPE).requires_grad_(True)
    xR = torch.full((Nbc, 1), xMax, device=device, dtype=DTYPE).requires_grad_(True)

    uL = model(xL, t)
    uR = model(xR, t)

    uxL = torch.autograd.grad(uL, xL, torch.ones_like(uL), create_graph=True)[0]
    uxR = torch.autograd.grad(uR, xR, torch.ones_like(uR), create_graph=True)[0]

    return torch.mean(((uL - uR)**2) + (uxL - uxR)**2)


def invariants_of(model, attached):
    """
    I_i(t_j) for every active invariant, on the DEDICATED quadrature grid.

    Deliberately *not* the collocation grid. The two answer different questions
    and have unrelated resolution requirements:

      * the collocation grid is sized from the parameter count (`fac`) -- an
        optimisation-theoretic choice about how many residual equations to impose;
      * the quadrature grid has to resolve the invariant *density*, which is
        quadratic and cubic in (u, u_x) and so spreads to far higher harmonics
        than u itself once the solution steepens.

    Tying them together was silently fatal, and is what drove training to the
    frozen field. For CH the h1 density starts at harmonic 4 and reaches 128 by
    t=5; on the Nx=45 grid (Nyquist 22) about 2.4% of its energy is aliased and
    the quadrature reports that as ~5e-2 of drift -- two orders of magnitude
    above the tolerance being demanded. Worse, the aliasing is *time-dependent*
    (identically zero at t=0), so the one way to make it cancel in I(t) - I(0)
    is for u not to depend on t. The frozen field u = u_0 was therefore the
    essentially unique feasible point. `quadrature_floor` now checks for this.

    `attached=False` is only right when the invariants are merely monitored;
    when they drive a constraint they must stay on the graph.
    """
    xq = quad_inputs[:, 0:1].detach().requires_grad_(True)
    tq = quad_inputs[:, 1:2].detach()
    u = model(xq, tq)
    u_x = torch.autograd.grad(u, xq, torch.ones_like(u), create_graph=attached)[0]
    if not attached:
        u, u_x = u.detach(), u_x.detach()
    ug, uxg = u.reshape(Nt, NX_QUAD), u_x.reshape(Nt, NX_QUAD)
    return {n: _quad(_DENSITIES[n](ug, uxg), dx_quad) for n in ACTIVE_INVARIANTS}


def compute_losses(inputs, model, epoch, no_grad_H = True):
    """
    Computes:
      - PDE residual loss in H1 norm (L2 + λ·‖r_x‖²)
      - Initial condition MSE loss
      - Periodic BC loss
      - Hamiltonian H(t) per time step (detached, for monitoring only)
    """
    x = inputs[:, 0:1].detach().requires_grad_(True)
    t = inputs[:, 1:2].detach().requires_grad_(True)

    u = model(x, t)
    ones_u = torch.ones_like(u)

    # First derivatives
    u_x, u_t = torch.autograd.grad(u, [x, t], ones_u, create_graph=True)

    # Second and third spatial derivatives
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]

    # Mixed derivative u_xxt
    u_xxt = torch.autograd.grad(u_xx, t, torch.ones_like(u_xx), create_graph=True)[0]

    r = u_t - u_xxt + 3.0 * u * u_x - 2.0 * u_x * u_xx - u * u_xxx

    lam = lambda_grad_weight(epoch)
    pde_L2 = torch.mean(r**2)

    if lam > 0.0:
        r_x = torch.autograd.grad(r, x, torch.ones_like(r), create_graph=True)[0]
        pde_H1 = pde_L2 + lam * torch.mean(r_x**2)
    else:
        pde_H1 = pde_L2

    # IC loss: u(x, 0) = u_0(x)
    ic_mask = (t.detach().abs() < 1e-6).squeeze()
    x_ic = x[ic_mask].detach()
    u_ic_pred = model(x_ic, torch.zeros_like(x_ic))
    ic_loss = torch.mean((u_ic_pred - u_0(x_ic))**2)

    # Periodic BC loss
    bc_loss = periodic_bc_loss(model)

    # Conserved quantities I_i(t), one value per collocation time -- computed on
    # their own, finer x-grid rather than from `u`/`u_x` above. See `invariants_of`.
    inv_vals = invariants_of(model, attached=not no_grad_H)

    return pde_H1, ic_loss, bc_loss, inv_vals


# ── Training setup ────────────────────────────────────────────────────────────

H_QUAD_RULE = 'boole'   # 'boole' | 'rect'
MAX_H = False            # collapse each invariant's Nt constraints into their max

# A condition that holds by construction needs neither a penalty nor a dual.
CONSTRAIN_IC = not HARD_IC
CONSTRAIN_BC = not HARD_BC

# ── Invariant quadrature grid (decoupled from the collocation grid) ──────────
#
# See `invariants_of` for why these must not be the same grid. Measured by
# pushing the EXACT reference solution through this script's own quadrature:
#
#   NX_QUAD    45        89        181       361
#   h1 drift   5.19e-02  1.05e-02  7.53e-04  4.00e-05
#
# so Nx=45 cannot measure CH conservation to better than ~5e-2, while the
# tolerance schedule tightens to 1e-4. NX_QUAD=361 clears that floor by 2.5x.
NX_QUAD = 361
NX_QUAD += (-(NX_QUAD - 1)) % 4   # Boole needs (NX_QUAD - 1) % 4 == 0, as for Nx
QUAD_FLOOR_CHECK = True           # abort if a tolerance floor is below the quadrature floor

# ── Invariants to constrain ───────────────────────────────────────────────────
#
# Selects from the Camassa-Holm hierarchy, in the order the dual groups are
# created. Constraining the low-order members alongside the Hamiltonian costs
# one extra quadrature each and is far better conditioned: I1 is linear in the
# network output, I2 quadratic, I3 cubic.
ACTIVE_INVARIANTS = ('mass', 'h1', 'h2')

_DENSITIES = {'mass': ch_mass, 'h1': ch_h1, 'h2': ch_density}
_EPS_FLOOR = {'mass': 1e-2, 'h1': 1e-1, 'h2': 1e-1}
_DUAL_LR   = {'mass': 5e-1, 'h1': 5e-1, 'h2': 5e-1}

# Tolerance scale S_i. 'l1' uses int|density|dx, 'value' uses |I^0|. They agree
# exactly for every CH invariant here (u_0 = 0.2 + 0.1cos2x > 0, so all three
# densities are one-signed), but 'l1' is the safe default: |I^0| can vanish by
# cancellation for other initial data, and a vanishing scale silently turns the
# constraint into a demand for exact conservation.
EPS_SCALE = 'l1'

# With the IC hard-encoded the trivial-solution trap is gone (see PINN), so the
# invariant constraints no longer need to be held back. Raise this if running
# with HARD_IC = False.
GRACE = 100
IC_EPS = 1e-6
BC_EPS = 1e-6
ICBC_DUAL_LR = 5e-1

loss_type = 'cs'  # Chebyshev (max) aggregation

# ── Spectral reference solution (NOTES.md §5.8) ───────────────────────────────
#
# Every other diagnostic here is self-referential: the residual and the
# invariant drift are both computed from the network's own output, so a wrong
# field can score well on them. The frozen field u(x,t) = u_0(x) conserves every
# invariant *exactly* and is completely wrong; nothing above can tell.
#
# Camassa-Holm is integrated in momentum form, m_t = -(u m_x + 2 u_x m) with
# u = A^{-1} m, pseudospectrally on a periodic grid. There is no stiff linear
# term -- A^{-1} = (1 - d_xx)^{-1} is smoothing, not stiffening -- so explicit
# RK4 with a transport-CFL step is both stable and cheap.
#
# The solver checks itself against the same three invariants the constraints
# use: a reference that cannot conserve them is not fit to be a reference, and
# the run aborts rather than reporting errors against a bad baseline.

REF_ON        = True    # compute the reference and report true L2 errors
REF_NX        = 256     # spectral modes; periodic grid, right endpoint excluded
REF_NT        = 101     # time slices at which the error is evaluated
REF_CFL       = 0.05    # substep from dt = REF_CFL * dx_ref / max|u|
REF_EVERY     = 25      # evaluate the network against the reference every N epochs
REF_MAX_DRIFT = 1e-6    # abort if the reference's own invariant drift exceeds this


def _ref_ops(N, L):
    """Fourier differentiation and Helmholtz inverse on a periodic grid of N points."""
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)

    def D(f, n=1):
        return np.real(np.fft.ifft((1j * k) ** n * np.fft.fft(f)))

    def A_inv(f):                       # (1 - d_xx)^{-1}, i.e. m -> u
        return np.real(np.fft.ifft(np.fft.fft(f) / (1.0 + k ** 2)))

    return k, D, A_inv


def reference_solution():
    """Pseudospectral RK4 reference for CH. Returns (x, t, U) with U of shape (Nx, Nt)."""
    L = xMax - xMin
    xr = xMin + L * np.arange(REF_NX) / REF_NX      # periodic grid: excludes xMax
    tr = np.linspace(0.0, tMax, REF_NT)
    dx_r = L / REF_NX
    _, D, A_inv = _ref_ops(REF_NX, L)

    u0 = u_0(torch.as_tensor(xr, dtype=torch.float64)).numpy()

    def rhs(m):
        u = A_inv(m)
        return -(u * D(m) + 2.0 * D(u) * m)

    dt_slice = tr[1] - tr[0]
    sub = max(1, int(np.ceil(dt_slice / (REF_CFL * dx_r / max(np.abs(u0).max(), 1e-12)))))
    h = dt_slice / sub

    U = np.empty((REF_NX, REF_NT))
    U[:, 0] = u0
    m = u0 - D(u0, 2)
    for j in range(1, REF_NT):
        for _ in range(sub):
            a = rhs(m); b = rhs(m + h * a / 2)
            c = rhs(m + h * b / 2); d = rhs(m + h * c)
            m = m + h * (a + 2 * b + 2 * c + d) / 6
        U[:, j] = A_inv(m)

    # Self-check, against the script's own density functions so the two cannot
    # disagree. Scaled by the L1 norm of the density, not |I^0|, for the reason
    # given at EPS_SCALE: |I^0| can vanish by cancellation.
    Ux = np.stack([D(U[:, j]) for j in range(REF_NT)], axis=1)
    drift = {}
    for n, f in _DENSITIES.items():
        I = np.array([f(torch.as_tensor(U[:, j]), torch.as_tensor(Ux[:, j])).numpy().sum() * dx_r
                      for j in range(REF_NT)])
        S = np.abs(f(torch.as_tensor(U[:, 0]), torch.as_tensor(Ux[:, 0])).numpy()).sum() * dx_r
        drift[n] = np.abs(I - I[0]).max() / (S + 1e-30)

    print(f"Reference  : spectral RK4, N={REF_NX}, {REF_NT} slices, "
          f"{sub} substeps/slice (dt={h:.2e})")
    print("             own invariant drift: "
          + ", ".join(f"{n}={d:.2e}" for n, d in drift.items()))
    worst = max(drift.values())
    if worst > REF_MAX_DRIFT:
        raise SystemExit(
            f"Reference conserves its own invariants only to {worst:.2e} "
            f"(> REF_MAX_DRIFT={REF_MAX_DRIFT:.1e}), so it cannot serve as a "
            f"baseline. Raise REF_NX or lower REF_CFL.")
    return xr, tr, U


_REF_GRID = None    # (x, t) collocation of the reference grid, built once


def ref_errors(model, refsol):
    """Relative L2 error of u_theta against the reference: global, per slice, and raw."""
    global _REF_GRID
    xr, tr, U = refsol
    if _REF_GRID is None:
        Xg, Tg = np.meshgrid(xr, tr, indexing='ij')
        _REF_GRID = (torch.tensor(Xg.ravel()[:, None], device=device, dtype=DTYPE),
                     torch.tensor(Tg.ravel()[:, None], device=device, dtype=DTYPE))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        Up = model(*_REF_GRID).cpu().numpy().reshape(U.shape)
    if was_training:
        model.train()
    err = Up - U
    per_slice = np.linalg.norm(err, axis=0) / (np.linalg.norm(U, axis=0) + 1e-30)
    return np.linalg.norm(err) / np.linalg.norm(U), per_slice, err


# ── Dual group layout ─────────────────────────────────────────────────────────
#
# One entry per dual group, in registration order. `build_constraints` walks
# this same list, so the two cannot drift -- which is the failure mode the
# earlier hand-written offsets had (a disabled IC group silently shifted the BC
# group's values onto the wrong duals).
m_inv = 1 if MAX_H else Nt

GROUPS = [SimpleNamespace(kind='inv', name=n, m=m_inv,
                          eps_floor=_EPS_FLOOR[n], lr=_DUAL_LR[n])
          for n in ACTIVE_INVARIANTS]
if CONSTRAIN_IC:
    GROUPS.append(SimpleNamespace(kind='cond', name='ic', m=1,
                                  eps_floor=IC_EPS, lr=ICBC_DUAL_LR))
if CONSTRAIN_BC:
    GROUPS.append(SimpleNamespace(kind='cond', name='bc', m=1,
                                  eps_floor=BC_EPS, lr=ICBC_DUAL_LR))

model = PINN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters : {n_params} (Ntheta from Nx_from_arch: {Ntheta})")
print(f"Grid       : Nx={Nx}, Nt={Nt}, collocation points={Nx * Nt}")
print(f"Encoding   : hard IC={HARD_IC} (tau={IC_TAU}), "
      f"hard BC={HARD_BC} (K={N_MODES} modes), input dim={D_IN}")

REFSOL = reference_solution() if REF_ON else None
if REFSOL is not None:
    # The score of the frozen field u(x,t) = u_0(x). It conserves every
    # invariant exactly, so it is the number a run lands on when the
    # constraints are satisfied and no dynamics have been learned -- a
    # tripwire, not a target.
    REF_FROZEN = (np.linalg.norm(REFSOL[2] - REFSOL[2][:, :1])
                  / np.linalg.norm(REFSOL[2]))
    print(f"             frozen field u=u_0 scores rel. L2 = {REF_FROZEN:.4f}")
else:
    REF_FROZEN = float('nan')

# SoftAdaptive weights (updated each epoch)
loss_components = 3
lambdas = torch.ones(loss_components, device=device, dtype=DTYPE) / loss_components

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, foreach=True)

# # Replicates Keras ExponentialDecay: lr(t) = lr_0 * 0.9^(t/epochs)
# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda step: 0.5 ** (step / epochs)
# )

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9**(1/epochs))

# Build meshgrid collocation inputs: shape (Nx*Nt, 2)
x_np = np.linspace(xMin, xMax, Nx, dtype=np.float32)
t_np = np.linspace(0.0, tMax, Nt, dtype=np.float32)
x_grid_np, t_grid_np = np.meshgrid(x_np, t_np)
inputs = torch.tensor(
    np.column_stack([x_grid_np.ravel(), t_grid_np.ravel()]),
    device=device, dtype=DTYPE,
)

# Invariant quadrature grid: a finer x-grid, at the same collocation times.
x_quad_np = np.linspace(xMin, xMax, NX_QUAD, dtype=np.float32)
dx_quad = (xMax - xMin) / (NX_QUAD - 1)
_xq_np, _tq_np = np.meshgrid(x_quad_np, t_np)
quad_inputs = torch.tensor(
    np.column_stack([_xq_np.ravel(), _tq_np.ravel()]),
    device=device, dtype=DTYPE,
)
print(f"Quadrature : NX_QUAD={NX_QUAD} (dx={dx_quad:.6f}) -> {NX_QUAD * Nt} points "
      f"for the invariants, vs {Nx * Nt} for the residual")

_quad = quad_boole if H_QUAD_RULE == 'boole' else quad_rect

# Reference value I^0 and tolerance scale S for each invariant, from the exact
# initial condition and the same quadrature the constraint uses.
#
# S = int |density| dx rather than |I^0|. The two coincide whenever the density
# has one sign -- true of every CH invariant here, since u_0 = 0.2 + 0.1cos2x
# is strictly positive -- but |I^0| can vanish by cancellation (int cos = 0
# exactly, for the KdV initial condition), and a zero scale would silently
# demand exact conservation. The L1 norm of the density never vanishes for a
# nontrivial field.
# On the QUADRATURE grid, with the quadrature spacing, so the systematic rule
# error still cancels exactly in I(t) - I^0 at t = 0.
x_1d = torch.tensor(x_quad_np, device=device, dtype=DTYPE)
_u0, _u0x = u_0(x_1d), u_0_x(x_1d)
REF = {}
for n in ACTIVE_INVARIANTS:
    _f0 = _DENSITIES[n](_u0, _u0x)
    _I0 = _quad(_f0, dx_quad)
    _L1 = _quad(_f0.abs(), dx_quad)
    _S = _L1 if EPS_SCALE == 'l1' else torch.abs(_I0)
    # Relative degeneracy test: an exactly-zero integral does not come back as
    # zero in float32, so an `<= 0` test would pass a vanishing scale through
    # and the constraint would silently demand conservation to round-off. Every
    # CH density here is one-signed, so |I0| == L1 and neither branch triggers;
    # the guard matters if u_0 is changed to something that cancels.
    if _S.item() <= 1e-6 * _L1.item():
        raise SystemExit(
            f"invariant {n!r}: tolerance scale {_S.item():.3e} is degenerate "
            f"against the density L1 norm {_L1.item():.3e} "
            f"(EPS_SCALE={EPS_SCALE!r}) -- use EPS_SCALE='l1'")
    REF[n] = (_I0, _S)
    print(f"Invariant {n:<5}: I0={_I0.item():+.6e}  scale={_S.item():.6e}")

# ── Quadrature floor: is the tolerance we demand even measurable? ────────────
#
# The failure this guards against is not hypothetical -- it is what made the
# constrained runs converge to the frozen field. See `invariants_of`.
#
# Push the EXACT reference solution through this script's own quadrature and
# ask what drift it reports. The true answer is ~1e-10 (the reference conserves
# to solver precision, see `reference_solution`), so whatever comes back is
# pure quadrature error. Any tolerance below it is unmeasurable, and because
# the error is time-dependent while a t-independent field's cancels exactly,
# demanding it hands the optimiser the frozen field as its only feasible point.

def quadrature_floor(refsol):
    """Drift this script's quadrature reports for the exact reference solution."""
    xr, tr, Ur = refsol
    N, L = len(xr), xMax - xMin
    kk = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    Uxr = np.real(np.fft.ifft((1j * kk)[:, None] * np.fft.fft(Ur, axis=0), axis=0))
    # exact trigonometric interpolation onto the quadrature grid
    ang = np.exp(1j * np.outer(x_quad_np.astype(np.float64) - xr[0], kk))
    Uq = np.real(ang @ (np.fft.fft(Ur, axis=0) / N))
    Uxq = np.real(ang @ (np.fft.fft(Uxr, axis=0) / N))
    out = {}
    for n in ACTIVE_INVARIANTS:
        f = _DENSITIES[n](torch.as_tensor(Uq.T), torch.as_tensor(Uxq.T))
        I = _quad(f, dx_quad)
        out[n] = ((I - I[0]).abs().max() / REF[n][1].item()).item()
    return out


if QUAD_FLOOR_CHECK and REFSOL is not None:
    _floor = quadrature_floor(REFSOL)
    print("Quad floor : " + ", ".join(f"{n}={v:.2e}" for n, v in _floor.items())
          + "   (drift the quadrature reports for the EXACT solution)")
    _bad = [(n, v) for n, v in _floor.items() if v > _EPS_FLOOR[n]]
    if _bad:
        raise SystemExit(
            "Tolerance below the quadrature floor for "
            + ", ".join(f"{n}: floor {v:.2e} > eps {_EPS_FLOOR[n]:.2e}" for n, v in _bad)
            + f".\nAt NX_QUAD={NX_QUAD} the constraint cannot resolve the conservation it "
              "demands, so the only feasible field is one independent of t (the frozen\n"
              "field u = u_0). Raise NX_QUAD, or raise the affected _EPS_FLOOR entries "
              "above the floor.")
elif QUAD_FLOOR_CHECK:
    print("Quad floor : SKIPPED (REF_ON=False) -- tolerances are unverified")


from humancompatible.train.dual_optim import nuPI, iALM, ALM

if CONSTRAINED:
    if not GROUPS:
        raise SystemExit("CONSTRAINED=True but no constraint group is enabled")
    # dual_opt = nuPI(m=GROUPS[0].m, nu=0, ki=0.1, kp=0.1, is_ineq=True, device=device)
    dual_opt = ALM(m=GROUPS[0].m, lr=GROUPS[0].lr, device=device, penalty=0.1,
                   is_ineq=True, momentum=0., dual_range=(0, 10000), augmentation='hpr', restart=True)
    # dual_opt = iALM(m=GROUPS[0].m, beta=0.05, sigma=1.001, gamma=1., dual_range=(-10.,10000.), is_ineq=True)
    for g in GROUPS[1:]:
        dual_opt.add_constraint_group(m=g.m, lr=g.lr, is_ineq=True)
    print("Dual groups: " + ", ".join(f"{g.name}(m={g.m}, lr={g.lr:g})" for g in GROUPS))

# Check the hard-encoded conditions on the untrained network: with the ansatz in
# place both residuals should sit at float32 round-off, not merely be small.
with torch.no_grad():
    _xc = torch.tensor(np.linspace(xMin, xMax, 501, dtype=np.float32),
                       device=device, dtype=DTYPE).reshape(-1, 1)
    _tc = torch.rand(501, 1, device=device, dtype=DTYPE) * tMax
    _ic_res = (model(_xc, torch.zeros_like(_xc)) - u_0(_xc)).abs().max().item()
    _bc_res = (model(torch.full_like(_tc, xMin), _tc)
               - model(torch.full_like(_tc, xMax), _tc)).abs().max().item()
# Reported relative to ||u_0||_inf: under the hard ansatz phi(0) is exactly 0,
# so the only residual left is a 1-ulp difference from evaluating u_0 on the
# strided view xt[:, 0:1] rather than on a contiguous tensor. Anything above a
# few float32 eps means the ansatz is not doing what it claims.
_u0_scale = u_0(_xc).abs().max().item() + 1e-30
print(f"Check      : max|u(x,0)-u_0(x)|={_ic_res:.3e} "
      f"(rel {_ic_res/_u0_scale:.1e}){'  [hard]' if HARD_IC else ''},  "
      f"max|u(xMin,t)-u(xMax,t)|={_bc_res:.3e}"
      f"{'  [hard]' if HARD_BC else ''}")


# ── Constraint / objective assembly ───────────────────────────────────────────

def eps_of(g, epoch):
    """
    Relative tolerance for group `g` at `epoch`.

    Invariant groups follow the continuation schedule eps = max(1/(epoch-grace),
    floor): loose when the constraint first switches on, tightening toward
    `floor`. Condition groups (IC / BC, only present when not hard-encoded)
    carry a fixed absolute tolerance instead.
    """
    if g.kind != 'inv':
        return g.eps_floor
    if epoch <= GRACE:
        return float('inf')          # inactive
    return max(1.0 / (np.sqrt(epoch) - GRACE), g.eps_floor)


def build_constraints(inv_vals, ic_loss, bc_loss, epoch):
    """
    One constraint tensor per dual group, in GROUPS order, plus a dict of raw
    values for logging: report[name] = (drift/S, drift/|I^0|).

    Invariant groups impose, for each collocation time t_j,

        c_i(t_j) = |I_i(t_j) - I_i^0| - eps_i * S_i  <=  0,

    i.e. a bound on the ABSOLUTE drift whose tolerance carries the scale S_i.
    That is the same feasible set as bounding the relative drift
    |I_i - I_i^0| / S_i, but the gradient is not amplified by 1/S_i. Do NOT also
    divide the drift by S_i -- the tolerance already carries it.

    Blocks keep their full width (m = 1 under MAX_H, else Nt) even while a group
    is inactive during the grace period, so the list arity is invariant across
    every flag combination; zeros leave both the dual update and the Lagrangian
    term untouched.
    """
    blocks, report = [], {}

    for g in GROUPS:
        if g.kind == 'inv':
            I0 = REF[g.name][0] if USE_TRUE_H0 else inv_vals[g.name][0]
            S = REF[g.name][1]
            drift = torch.abs(inv_vals[g.name] - I0)          # (Nt,)
            aI0 = torch.abs(I0)
            # drift/S is what the constraint bounds; drift/|I^0| is the
            # conventional relative error, logged alongside for comparability
            # (NaN where I^0 vanishes -- see EPS_SCALE).
            report[g.name] = (
                (drift / S).max().item(),
                # Relative guard, as for the scale itself: |I0| = 4.8e-8 for the
                # KdV mass is numerically zero, and an `> 0` test let it through
                # and printed a meaningless |dI|/|I0| = 6e+05.
                (drift / aI0).max().item()
                if aI0.item() > 1e-6 * S.item() else float('nan'),
            )

            eps = eps_of(g, epoch)
            c = torch.zeros(Nt, device=device, dtype=DTYPE) if eps == float('inf') \
                else drift - eps * S
            blocks.append(c.max().unsqueeze(0) if g.m == 1 else c)
        else:
            val = ic_loss if g.name == 'ic' else bc_loss
            report[g.name] = (val.item(), float('nan'))
            blocks.append(val.unsqueeze(0) - g.eps_floor)

    return blocks, report


def build_objective(pde_loss, ic_loss, bc_loss):
    """
    Primal objective: the PDE residual, plus any condition that is neither
    hard-encoded nor constrained (Chebyshev max aggregation, as in the soft
    formulation). With HARD_IC and HARD_BC this is the PDE residual alone --

        min_theta  ||r(theta)||_{H1}^2   s.t.  the invariant drift constraints.
    """
    terms, weights = [pde_loss], [lambdas[0]]
    if not (HARD_IC or CONSTRAIN_IC):
        terms.append(ic_loss); weights.append(lambdas[1])
    if not (HARD_BC or CONSTRAIN_BC):
        terms.append(bc_loss); weights.append(lambdas[2])
    if len(terms) == 1:
        return pde_loss
    return torch.max(torch.stack(weights) * torch.stack(terms))

# ── Solution plotting (surface + IC / BC traces) ──────────────────────────────

def _eval_surface(model, N=600):
    """u_theta on a square (N+1)^2 grid. Returns U indexed as U[x_index, t_index]."""
    tspace = np.linspace(0.0, tMax, N + 1, dtype=np.float32)
    xspace = np.linspace(xMin, xMax, N + 1, dtype=np.float32)
    T_plot, X_plot = np.meshgrid(tspace, xspace)
    XT = torch.tensor(
        np.column_stack([X_plot.ravel(), T_plot.ravel()]), device=device, dtype=DTYPE
    )
    with torch.no_grad():
        u = model(XT[:, 0:1], XT[:, 1:2])
    return xspace, tspace, X_plot, T_plot, u.cpu().numpy().reshape(N + 1, N + 1)


def _boundary_u_x(model, tspace):
    """u_x at x=xMin and x=xMax, to check periodicity of the derivative."""
    t = torch.tensor(tspace, device=device, dtype=DTYPE).reshape(-1, 1)
    out = []
    for xb in (xMin, xMax):
        x = torch.full_like(t, xb).requires_grad_(True)
        u = model(x, t)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u))[0]
        out.append(u_x.detach().cpu().numpy().ravel())
    return out


def plot_solution(model, tag='', surf_path=None, edge_path=None):
    """
    Two figures:
      1. 3-D surface with the IC (exact vs learned) and both boundary traces on it
      2. 1-D panels: IC fit, BC value periodicity, BC derivative periodicity
    """
    xspace, tspace, X_plot, T_plot, U = _eval_surface(model)
    u0_exact = u_0(torch.tensor(xspace, device=device, dtype=DTYPE)).cpu().numpy()
    uxL, uxR = _boundary_u_x(model, tspace)

    zeros_x = np.zeros_like(xspace)
    ic_err = np.abs(U[:, 0] - u0_exact).max()
    bc_err = np.abs(U[0, :] - U[-1, :]).max()
    bcx_err = np.abs(uxL - uxR).max()

    # --- 3-D surface with IC/BC curves overlaid ---
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_plot, T_plot, U, cmap='viridis', alpha=0.6)
    ax.plot(xspace, zeros_x, u0_exact, 'r--', lw=2.5, label='IC exact $u_0(x)$')
    ax.plot(xspace, zeros_x, U[:, 0], 'k-', lw=1.5, label='IC learned $u_\\theta(x,0)$')
    ax.plot(np.full_like(tspace, xMin), tspace, U[0, :],  'b-', lw=1.5, label='BC $x=x_{\\min}$')
    ax.plot(np.full_like(tspace, xMax), tspace, U[-1, :], 'm-', lw=1.5, label='BC $x=x_{\\max}$')
    ax.view_init(35, 35)
    ax.set_xlabel('$x$'); ax.set_ylabel('$t$'); ax.set_zlabel('$u_\\theta(x,t)$')
    ax.set_title(f'Solution to Camassa-Holm equation{tag}')
    ax.set_box_aspect(None, zoom=0.85)
    ax.legend(loc='upper left', fontsize=8)
    if surf_path is not None and save_fig:
        plt.savefig(surf_path, dpi=300)
    plt.show()
    plt.close(fig)

    # --- 1-D IC / BC diagnostics ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(xspace, u0_exact, 'r--', lw=2, label='exact $u_0(x)$')
    axes[0].plot(xspace, U[:, 0], 'k-', lw=1.2, label='$u_\\theta(x,0)$')
    axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$u$')
    axes[0].set_title(f'Initial condition (max err {ic_err:.2e})')

    axes[1].plot(tspace, U[0, :],  'b-', lw=1.2, label='$u_\\theta(x_{\\min},t)$')
    axes[1].plot(tspace, U[-1, :], 'm--', lw=1.2, label='$u_\\theta(x_{\\max},t)$')
    axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$u$')
    axes[1].set_title(f'BC: value periodicity (max err {bc_err:.2e})')

    axes[2].plot(tspace, uxL, 'b-', lw=1.2, label='$\\partial_x u_\\theta(x_{\\min},t)$')
    axes[2].plot(tspace, uxR, 'm--', lw=1.2, label='$\\partial_x u_\\theta(x_{\\max},t)$')
    axes[2].set_xlabel('$t$'); axes[2].set_ylabel('$u_x$')
    axes[2].set_title(f'BC: derivative periodicity (max err {bcx_err:.2e})')

    for a in axes:
        a.legend(fontsize=8); a.grid()
    fig.suptitle(f'Initial and boundary conditions{tag}')
    fig.tight_layout()
    if edge_path is not None and save_fig:
        plt.savefig(edge_path, dpi=300)
    plt.show()
    plt.close(fig)


# ── History buffers ───────────────────────────────────────────────────────────

# H_* keys track one designated invariant so the existing plots and CSV columns
# keep their meaning; the Hamiltonian if it is active, else the first group.
H_KEY = 'h2' if any(g.name == 'h2' for g in GROUPS) else (
    GROUPS[0].name if GROUPS and GROUPS[0].kind == 'inv' else None)

hist = {k: [] for k in ['loss', 'pde', 'ic', 'bc', 'ref_l2',
                        'H_min', 'H_max', 'H_mean', 'H_std',
                        'H_abs_err', 'H_rel_err']}
ref_l2_last = float('nan')
for g in GROUPS:
    hist[f'drift_{g.name}'] = []
    hist[f'reldrift_{g.name}'] = []
    hist[f'dual_{g.name}'] = []

t_start = time()
for epoch in range(epochs):
    # if epoch % 500 == 0 or epoch == epochs - 1:
    #     model.eval()
    #     plot_solution(model, tag=f' (epoch {epoch})')

    model.train()
    optimizer.zero_grad()

    pde_loss, ic_loss, bc_loss, inv_vals = compute_losses(
        inputs, model, epoch, no_grad_H=not CONSTRAINED)

    if CONSTRAINED:
        loss = build_objective(pde_loss, ic_loss, bc_loss)
        blocks, report = build_constraints(inv_vals, ic_loss, bc_loss, epoch)
        # Passing the per-group list (not a flat tensor) makes dual_optim verify
        # each group's arity, so a layout slip raises instead of feeding one
        # group's values to another.
        lagrangian = dual_opt.forward_update(loss, blocks)
        lagrangian.backward()
    else:
        # Chebyshev (max) loss aggregation over whatever is not hard-encoded
        stack = [pde_loss] + ([] if HARD_IC else [ic_loss]) + ([] if HARD_BC else [bc_loss])
        loss = torch.max(lambdas[:len(stack)] * torch.stack(stack))
        _, report = build_constraints(inv_vals, ic_loss, bc_loss, epoch)
        loss.backward()

    optimizer.step()
    scheduler.step()

    model.eval()
    hist['loss'].append(loss.item())
    hist['pde'].append(pde_loss.item())
    hist['ic'].append(ic_loss.item())
    hist['bc'].append(bc_loss.item())

    # True error against the reference. Every REF_EVERY epochs rather than
    # every epoch; nan elsewhere so the column stays one row per epoch.
    if REFSOL is not None and (epoch % REF_EVERY == 0 or epoch == epochs - 1):
        ref_l2_last = ref_errors(model, REFSOL)[0]
        hist['ref_l2'].append(ref_l2_last)
    else:
        hist['ref_l2'].append(float('nan'))

    with torch.no_grad():
        for i, g in enumerate(GROUPS):
            hist[f'drift_{g.name}'].append(report[g.name][0])
            hist[f'reldrift_{g.name}'].append(report[g.name][1])
            hist[f'dual_{g.name}'].append(
                dual_opt.param_groups[i]['params'][0].max().item()
                if CONSTRAINED else 0.0)

        if H_KEY is not None:
            H_val = inv_vals[H_KEY]
            I0 = REF[H_KEY][0]
            hist['H_min'].append(H_val.min().item())
            hist['H_max'].append(H_val.max().item())
            hist['H_mean'].append(H_val.mean().item())
            # correction=0: population std, matching tf.math.reduce_std in the
            # original. torch defaults to the sample std (correction=1).
            hist['H_std'].append(H_val.std(correction=0).item())
            H_abs = (H_val - I0).abs()
            hist['H_abs_err'].append(H_abs.max().item())
            hist['H_rel_err'].append((H_abs / (I0.abs() + 1e-16)).max().item())

    if epoch % 100 == 0 or epoch == epochs - 1:
        # Duals are read per group, so nothing depends on a flat offset.
        parts = ' '.join(
            f"{g.name}={hist[f'drift_{g.name}'][-1]:.2e}"
            f"(mu={hist[f'dual_{g.name}'][-1]:.2e})" for g in GROUPS)
        print(f"Epoch {epoch+1:5d}/{epochs} | obj={loss.item():.4e} | "
              f"IC={ic_loss.item():.2e} BC={bc_loss.item():.2e} | {parts}"
              + (f" | relL2={ref_l2_last:.3e}" if REFSOL is not None else ""))

print(f"\nLoss type             : {loss_type}")
print(f"IC MSE (diagnostic)   : {hist['ic'][-1]:.4e}"
      f"{'  [hard-encoded]' if HARD_IC else ''}")
print(f"BC MSE (diagnostic)   : {hist['bc'][-1]:.4e}"
      f"{'  [hard-encoded]' if HARD_BC else ''}")
if H_KEY is not None:
    print(f"Hamiltonian mean      : {hist['H_mean'][-1]:.6f}")
    print(f"Hamiltonian std       : {hist['H_std'][-1]:.6f}")
    print(f"Hamiltonian max       : {hist['H_max'][-1]:.6f}")
    print(f"Hamiltonian min       : {hist['H_min'][-1]:.6f}")
    print(f"H relative error      : {hist['H_rel_err'][-1]:.4e}")
if CONSTRAINED and GROUPS:
    print("\nFinal constraint status (drift/scale vs tolerance, <=0 satisfied):")
    for g in GROUPS:
        eps = eps_of(g, epochs - 1)
        d = hist[f'drift_{g.name}'][-1]
        extra = ('' if g.kind != 'inv'
                 else f"   [|dI|/|I0| = {hist[f'reldrift_{g.name}'][-1]:.3e}]")
        print(f"  {g.name:<5}: {d:.4e} vs {eps:.4e}  -> {d - eps:+.3e}"
              f"   (mu={hist[f'dual_{g.name}'][-1]:.3e}){extra}")
if REFSOL is not None:
    REF_L2, REF_PER_SLICE, REF_ERR = ref_errors(model, REFSOL)
    print(f"\nRel. L2 vs reference  : {REF_L2:.4e}")
    print(f"  frozen field u=u_0  : {REF_FROZEN:.4e}"
          f"   -> {'NO dynamics learned' if REF_L2 > 0.95 * REF_FROZEN else 'better than frozen'}")
    print(f"  worst time slice    : {REF_PER_SLICE.max():.4e} at t={REFSOL[1][REF_PER_SLICE.argmax()]:.2f}")
print(f"Training time         : {time() - t_start:.1f}s")

torch.save(model, f'./results/camassa/{"CONSTRAINED" if CONSTRAINED else "UNCONSTRAINED"}/model.pt')


# ── Plotting helpers ──────────────────────────────────────────────────────────

learning_rate_type = 'plateau'

def fig_path(tag):
    return f"./results/camassa/{'CONSTRAINED' if CONSTRAINED else 'UNCONSTRAINED'}/{tag}_epochs_{epochs}_lr_{learning_rate_type}_{loss_type}.png"


def save_show(path):
    if save_fig:
        plt.savefig(path, dpi=300)
    plt.show()


# Loss curves
plt.semilogy(hist['loss'],  label='Total Loss')
plt.semilogy(hist['pde'],   label='PDE Loss')
plt.semilogy(hist['ic'],    label='Initial Conditions Loss')
plt.semilogy(hist['bc'],    label='Boundary Conditions Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Loss Contributions'); plt.legend(); plt.grid()
save_show(fig_path('loss'))
plt.clf()

# Hamiltonian min/max
plt.plot(hist['H_min'], label='min H')
plt.plot(hist['H_max'], label='max H')
plt.xlabel('Epoch'); plt.ylabel('Hamiltonian')
plt.title('Hamiltonian over epochs'); plt.legend(); plt.grid()
save_show(fig_path('H_loss'))
plt.clf()

# Hamiltonian mean ± std
H_mean = np.array(hist['H_mean'])
H_std  = np.array(hist['H_std'])
plt.plot(H_mean)
plt.fill_between(range(len(H_mean)), H_mean - H_std, H_mean + H_std, alpha=0.2)
plt.xlabel('Epoch'); plt.ylabel('Hamiltonian mean')
plt.title('Hamiltonian mean over epochs with standard deviation'); plt.grid()
save_show(fig_path('H_loss_mean'))
plt.clf()

# Hamiltonian std
plt.plot(hist['H_std'])
plt.xlabel('Epoch'); plt.ylabel('Hamiltonian std')
plt.title('Hamiltonian standard deviation over epochs'); plt.grid()
save_show(fig_path('H_loss_std'))
plt.clf()

# Hamiltonian absolute error
plt.plot(hist['H_abs_err'])
plt.xlabel('Epoch'); plt.ylabel('Hamiltonian absolute error')
plt.title('Hamiltonian absolute error over epochs'); plt.grid()
save_show(fig_path('H_loss_abs_error'))
plt.clf()

# Hamiltonian relative error
plt.plot(hist['H_rel_err'])
plt.xlabel('Epoch'); plt.ylabel('Hamiltonian relative error')
plt.title('Hamiltonian relative error over epochs'); plt.grid()
save_show(fig_path('H_loss_rel_error'))
plt.clf()

# True L2 error against the spectral reference (NOTES.md §5.8)
if REFSOL is not None:
    _xr, _tr, _Ur = REFSOL
    _e = np.array(hist['ref_l2'])
    _k = np.isfinite(_e)
    plt.semilogy(np.arange(len(_e))[_k], _e[_k], label='rel. $L^2$ vs reference')
    plt.axhline(REF_FROZEN, ls='--', c='r',
                label=f'frozen $u_0$ ({REF_FROZEN:.3f})')
    plt.xlabel('Epoch'); plt.ylabel('relative $L^2$ error')
    plt.title('True error against the spectral reference')
    plt.legend(); plt.grid()
    save_show(fig_path('ref_l2'))
    plt.clf()

    # The reading of these panels: a travelling wave is *slanted* stripes. The
    # frozen field gives stripes parallel to the t axis, a collapse fades out.
    _Up = _Ur + REF_ERR
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # Shared scale set by the reference, so the two panels are comparable; the
    # network saturates where it leaves the reference's range.
    _lo, _hi = _Ur.min(), _Ur.max()
    for _a, _Z, _ttl in [(ax[0, 0], _Ur, 'reference $u$'),
                         (ax[0, 1], _Up, 'network $u_\\theta$')]:
        _im = _a.pcolormesh(_tr, _xr, _Z, cmap='viridis', vmin=_lo, vmax=_hi,
                            shading='auto')
        _a.set_xlabel('$t$'); _a.set_ylabel('$x$'); _a.set_title(_ttl)
        fig.colorbar(_im, ax=_a)
    _m = np.abs(REF_ERR).max()
    _im = ax[1, 0].pcolormesh(_tr, _xr, REF_ERR, cmap='RdBu_r',
                              vmin=-_m, vmax=_m, shading='auto')
    ax[1, 0].set_xlabel('$t$'); ax[1, 0].set_ylabel('$x$')
    ax[1, 0].set_title(f'$u_\\theta - u$  (max {_m:.2e})')
    fig.colorbar(_im, ax=ax[1, 0])
    ax[1, 1].semilogy(_tr, REF_PER_SLICE, label='network')
    ax[1, 1].axhline(REF_FROZEN, ls='--', c='r', label='frozen $u_0$ (global)')
    ax[1, 1].set_xlabel('$t$'); ax[1, 1].set_ylabel('relative $L^2$ error')
    ax[1, 1].set_title(f'Error per time slice (global {REF_L2:.3e})')
    ax[1, 1].legend(); ax[1, 1].grid()
    fig.suptitle('Camassa-Holm: network against the spectral reference')
    fig.tight_layout()
    if save_fig:
        fig.savefig(fig_path('ref_compare'), dpi=300)
    plt.show()
    plt.close(fig)

# 3-D solution surface, with the IC and BC traces drawn on it
model.eval()
plot_solution(model, surf_path=fig_path('sol'), edge_path=fig_path('ic_bc'))

# ── Save training history ─────────────────────────────────────────────────────

cols = {
    'total_loss':            hist['loss'],
    'pde_loss':              hist['pde'],
    'data_fitting_loss_0':   hist['ic'],
    'data_fitting_loss_l_r': hist['bc'],
}
if H_KEY is not None:
    cols.update({
        'H_loss_min':       hist['H_min'],
        'H_loss_max':       hist['H_max'],
        'H_loss_mean':      hist['H_mean'],
        'H_loss_std':       hist['H_std'],
        'H_loss_abs_error': hist['H_abs_err'],
        'H_loss_rel_error': hist['H_rel_err'],
    })
if REFSOL is not None:
    cols['ref_rel_l2'] = hist['ref_l2']
for _g in GROUPS:
    cols[f'drift_{_g.name}'] = hist[f'drift_{_g.name}']
    cols[f'reldrift_{_g.name}'] = hist[f'reldrift_{_g.name}']
    cols[f'dual_{_g.name}'] = hist[f'dual_{_g.name}']
df = pd.DataFrame(cols)
csv_path = (f'./results/camassa/{"CONSTRAINED" if CONSTRAINED else "UNCONSTRAINED"}'
            f'/training_history_pytorch.csv')
df.to_csv(csv_path, index=False)
print(f"Saved training history to {csv_path}")
