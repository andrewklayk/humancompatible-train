"""
PyTorch port of `kdv_trainableH1_tf.py` (TensorFlow / Keras), with an optional
constrained formulation on top.

With `CONSTRAINED = True` (the default) the initial condition, the periodic
boundary condition and the Hamiltonian drift become hard inequality constraints
handled by `humancompatible.train.dual_optim`, and the primal objective is the
PDE residual alone. Set `CONSTRAINED = False` to fall back to the faithful
translation described below -- a single weighted-sum loss with H monitored only.

--- Faithful-translation baseline (CONSTRAINED = False) ---

PINN for the KdV equation

    u_t - alpha (u^2)_x - rho u_x - nu u_xxx = 0

on a periodic domain, with the Hamiltonian H(t) = int V(u) - nu u_x^2 / 2 dx
monitored (not constrained) over training.

Deliberate deviations from the TF original, all of them behaviour-preserving or
explicitly marked:

  * a fixed random seed is set (the TF script seeds nothing);
  * `|z|^2` in the FFT helper is computed as `z.real^2 + z.imag^2`, which is the
    same value but avoids the non-differentiable point of complex `abs` at 0;
  * `H0` is hoisted out of the training loop (it is constant);
  * the final surface plot is extended with the IC / BC traces and a 1-D
    diagnostics panel, and is also drawn every `PLOT_EVERY` epochs during
    training (matching `CamassaHolm_trainableH1.py`);
  * figures and the history CSV both go to `./results/kdv/`;
  * dead code in the original (`lambda_grad`, `x_eval`/`t_eval`, the commented
    early-stopping test) is dropped;
  * BUG FIX, changes results: Boole's rule in `H` no longer double-weights the
    last node of the integration prefix (see `H`). Hamiltonian values are
    therefore NOT bit-comparable with TF runs; the old ones were too large by
    `(2*dx/45) * 7 * f[n1-1]`. Note `grad_L2_fft_batch` still carries a
    separate factor-of-Nx normalisation error from the original, left in place.

Everything else -- activation, initialiser, optimiser hyperparameters, learning
rate schedule, loss aggregation -- reproduces the TF script exactly.
"""

import argparse
import os
from time import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

DTYPE = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--constrained', action='store_true')
parser.add_argument('--true_h0', action='store_true')
args = parser.parse_args()
CONSTRAINED = args.constrained
USE_TRUE_H0 = args.true_h0      # reference I^0 from the exact IC; else I(t=0) of the net

# Deviation: the TF script seeds nothing. Fixed seed so runs are comparable.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

type_pde = 1  # 1 for KdV, 2 for the soliton variant  (`type` in the TF script)

if type_pde == 1:
    nu = -0.022**2
    alpha = -0.5
    rho = 0.
    xMin = 0.
    xMax = 2.
    tMax = 5.   # 10.
elif type_pde == 2:
    nu = -1.
    alpha = -3.
    rho = 0.
    xMin = -20.
    xMax = 20.
    tMax = 100.  # 4.

RESULTS_DIR = f'./results/kdv/{"CONSTRAINED" if CONSTRAINED else "UNCONSTRAINED"}'
os.makedirs(RESULTS_DIR, exist_ok=True)
save_fig = True

# ── Hard-encoded initial and boundary conditions (see PINNModel) ──────────────
#
# Declared here rather than with the rest of the training setup because the
# periodic embedding changes the network's input dimension, which feeds the
# parameter count the collocation grid is sized from.
HARD_IC = True     # u(.,0) = u_0 exactly, by construction
HARD_BC = True     # x-periodicity of u and all its derivatives, by construction
N_MODES = 8        # periodic feature modes K; network input dim becomes 2K+1
IC_TAU = 1.0       # phi(t) = 1 - exp(-t/tau); order of the time scale of u

D_IN = (2 * N_MODES + 1) if HARD_BC else 2


# ── Grid sizing from network architecture ─────────────────────────────────────

def Nx_from_arch(width, depth, fac=1.5, d_in=2, d_out=1):
    """
    Given a PINN architecture (width, depth) and an overparam factor fac,
    compute Nx = Nt such that Nx * Nt ~= N_params / fac, rounded up so that
    Boole's rule tiles the x-grid exactly.

    `H` needs (Nx - 1) % 4 == 0; on any other Nx the leftover nodes fall to a
    trapezoid tail. That tail spans a *fixed* number of intervals, so its
    absolute error is O(dx^3) and dominates -- the mixed rule converges at
    order 3, not Boole's 6. The plain sqrt gives Nx = 44 here, whose 3 leftover
    intervals put a 4.2e-2 relative error into H0 for type_pde=1 -- four times
    H_EPS_FLOOR, i.e. the constraint was tighter than the quadrature could
    measure. Rounding up to Nx = 45 makes the rule exact to machine precision.

    Returns (Nx, Nt, Ntheta, Ncoll_target). Note Nx * Nt then slightly exceeds
    Ncoll_target (2025 vs 1976 at the default), still far below Ntheta.
    """
    Ntheta = (d_in + 1) * width \
             + (depth - 1) * (width * width + width) \
             + d_out * (width + 1)

    Ncoll_target = int(Ntheta / fac)

    Nx = int(np.sqrt(Ncoll_target))
    Nx += (-(Nx - 1)) % 4        # next Nx with (Nx - 1) % 4 == 0
    Nt = Nx

    return Nx, Nt, Ntheta, Ncoll_target


def h_from_NxNt(Nx, Nt, xMin, xMax, tMax):
    """Grid spacings dx, dt and h = max(dx, dt)."""
    dx = (xMax - xMin) / (Nx - 1)
    dt = tMax / (Nt - 1)
    return dx, dt, max(dx, dt)


width = 80
depth = 4

Nx, Nt, Ntheta, Ncoll = Nx_from_arch(width=width, depth=depth, fac=10., d_in=D_IN)
dx, dt, h = h_from_NxNt(Nx, Nt, xMin, xMax, tMax)

# Loss-aggregation weights and the augmented-Chebyshev blend parameter. Both are
# non-trainable in the TF script (`tf.Variable(..., trainable=False)`), so they
# never enter the parameter list here either.
lambdas = torch.tensor([1., 1., 1.], dtype=DTYPE, device=device)
cheb_par = torch.tensor(0.5, dtype=DTYPE, device=device)
do_training = False  # gates the SoftAdapt update of `lambdas`; False in the original


# ── Physics ───────────────────────────────────────────────────────────────────

def u_0(x):
    if type_pde == 1:
        return torch.cos(np.pi * x)
    elif type_pde == 2:
        return 6. / (torch.cosh(x)**2)


def u_0_x(x):
    if type_pde == 1:
        return -np.pi * torch.sin(np.pi * x)
    elif type_pde == 2:
        return -12. * torch.sinh(x) / (torch.cosh(x)**3)


def V(u):
    return alpha * torch.pow(u, 3) / 3 + rho * torch.pow(u, 2) / 2


# ── Conserved densities: the KdV invariant hierarchy ──────────────────────────
#
# With w = alpha u^2 + rho u + nu u_xx = dH/du the equation is u_t = d/dx (dH/du),
# i.e. Hamiltonian with the Poisson operator d/dx. That structure gives three
# invariants on a periodic domain:
#
#     C0 = int u dx          (Casimir of d/dx -- linear in u)
#     C1 = int u^2/2 dx      (momentum -- quadratic)
#     H  = int (V(u) - nu u_x^2/2) dx
#
# dC1/dt = int u d_x w = -int u_x w = -int d_x(alpha u^3/3 + rho u^2/2
# + nu u_x^2/2) = 0, and dH/dt = int w d_x w = int d_x(w^2/2) = 0.
#
# `kdv_density` keeps its old name and normalisation so H values stay
# comparable across runs. The low-order members cost one quadrature each.

def kdv_mass(u, u_x):
    """C0 density."""
    return u


def kdv_mom(u, u_x):
    """C1 density (momentum)."""
    return 0.5 * torch.pow(u, 2)


def kdv_density(u, u_x):
    """Hamiltonian density."""
    return V(u) - nu * torch.pow(u_x, 2) / 2

def quad_boole(f, dx, axis=-1):
    """
    Boole's rule along `axis` for a uniform grid with spacing dx. Requires
    (N-1) % 4 == 0; otherwise Boole on the largest valid prefix and trapezoid
    on the remainder.

    DEVIATION FROM THE TF ORIGINAL (bug fix). The TF version slices block starts
    as `idx[0::4]` over arange(n1), which also picks up the *last* prefix node;
    that node is already in the block-end slice `idx[4::4]`, so it received
    weight 14 instead of the correct 7. Composite Boole gives weight 14 only to
    the shared *interior* block boundaries, 7 to the two outer endpoints. Using
    `idx[0:-1:4]` restricts the slice to genuine block starts and fixes it.

    The TF results are therefore no longer bit-reproducible: the old value was
    too large by exactly `(2*dx/45) * 7 * f[n1-1]`.
    """
    n = f.shape[axis]
    axis = axis % f.ndim

    def _trap_rem(rem):
        """Trapezoid rule over a contiguous tail segment."""
        return torch.sum(0.5 * (rem[..., 1:] + rem[..., :-1]), dim=-1) * dx

    # Degenerate
    if n <= 1:
        return torch.sum(f, dim=axis) * dx

    # Largest prefix with (n1 - 1) % 4 == 0
    n1 = n - ((n - 1) % 4)
    # Boole constant for uniform spacing
    c = (2.0 * dx) / 45.0

    idx_prefix = torch.arange(n1, device=f.device)
    f0 = torch.index_select(f, axis, idx_prefix[0:-1:4])  # block starts: 0,4,...,n1-5
    f1 = torch.index_select(f, axis, idx_prefix[1::4])    # 1,5,9,...
    f2 = torch.index_select(f, axis, idx_prefix[2::4])    # 2,6,10,...
    f3 = torch.index_select(f, axis, idx_prefix[3::4])    # 3,7,11,...
    f4 = torch.index_select(f, axis, idx_prefix[4::4])    # block ends: 4,8,...,n1-1

    # Boole's block weights per 5 nodes: [7, 32, 12, 32, 7]. Interior block
    # boundaries appear in both f0 and f4 and so total 14, as they should.
    s = 7.0 * torch.sum(f0, dim=axis)
    s += 32.0 * torch.sum(f1, dim=axis)
    s += 12.0 * torch.sum(f2, dim=axis)
    s += 32.0 * torch.sum(f3, dim=axis)
    s += 7.0 * torch.sum(f4, dim=axis)

    boole_part = c * s

    if n1 == n:
        return boole_part

    rem = torch.index_select(f, axis, torch.arange(n1 - 1, n, device=f.device))
    return boole_part + _trap_rem(rem)


def H(u, u_x, dx, density_fn=kdv_density, axis=-1):
    """Back-compatible wrapper: sample a density, then integrate it."""
    return quad_boole(density_fn(u, u_x), dx, axis)


# ── Spectral residual norms ───────────────────────────────────────────────────
#
# Both helpers use dx = L / Nx internally, which differs from the global
# dx = L / (Nx - 1). That is what the TF script does; preserved here.

def _wavenumbers(Nx, L, device):
    """TF's fftfreq-style wavenumber vector: [0..Nx//2, -Nx//2+1..-1] * 2 pi / L."""
    k_pos = torch.arange(0, Nx // 2 + 1, dtype=torch.float32, device=device)
    k_neg = torch.arange(-Nx // 2 + 1, 0, dtype=torch.float32, device=device)
    k = torch.cat([k_pos, k_neg], dim=0)
    return (2.0 * np.pi / L) * k


def grad_L2_fft_batch(r, L):
    """
    ||r_x||^2 per time slice, spectrally.

    r : (Nt, Nx)  ->  (Nt,)
    """
    Nx_local = r.shape[-1]
    k = _wavenumbers(Nx_local, L, r.device)

    r_hat = torch.fft.fft(r.to(torch.complex64), dim=-1)
    # |i k r_hat|^2 == k^2 |r_hat|^2, written via real/imag rather than abs() so
    # the backward pass stays finite at r_hat == 0.
    grad_energy = torch.sum((k**2) * (r_hat.real**2 + r_hat.imag**2), dim=-1)

    return grad_energy * (L / float(Nx_local))


def H1_norm_fft_batch(r, L):
    """
    ||r||_{H^1}^2 per time slice, spectrally. Unused by `custom_loss` (as in the
    TF script), kept as the counterpart to `grad_L2_fft_batch`.

    r : (Nt, Nx)  ->  (Nt,)
    """
    Nx_local = r.shape[-1]
    k = _wavenumbers(Nx_local, L, r.device)

    r_hat = torch.fft.fft(r.to(torch.complex64), dim=-1)
    weight = 1.0 + k**2

    H1_sq = torch.sum(weight * (r_hat.real**2 + r_hat.imag**2), dim=-1)

    return H1_sq * (L / float(Nx_local))


# ── Loss aggregation ──────────────────────────────────────────────────────────

def linear_loss_function(tensors, weights):
    """Weighted mean of the loss components (weights renormalised to sum to 1)."""
    stacked = torch.stack(tensors, dim=0)
    weights = weights / torch.sum(weights)
    return torch.sum(weights * stacked), 'ls'


def chebyshev_loss_function(tensors, weights):
    """Weighted max of the loss components."""
    stacked = torch.stack(tensors, dim=0)
    return torch.max(weights * stacked), 'cs'


def smooth_chebyshev_loss_function(mu, tensors, weights):
    """Log-sum-exp smoothing of the max."""
    weights = weights / torch.sum(weights)
    stacked = torch.stack(tensors, dim=0)
    exp_sum = torch.sum(torch.exp(stacked / mu), dim=0)
    return mu * torch.log(exp_sum), 'scs'


def augmented_chebyshev_loss_function(tensors, weights):
    """Convex blend of the Chebyshev and linear aggregations, weighted by cheb_par."""
    par = torch.sigmoid(cheb_par)  # in (0, 1)
    return (par * chebyshev_loss_function(tensors, weights)[0]
            + (1 - par) * linear_loss_function(tensors, weights)[0]), 'acs'


# ── Model ─────────────────────────────────────────────────────────────────────

def sigmoid_centered(x):
    return 2 * torch.sigmoid(x) - 1


class SigmoidCentered(nn.Module):
    def forward(self, x):
        return sigmoid_centered(x)


class PeriodicFeatures(nn.Module):
    """
    x  ->  [sin(2 pi k (x - xMin)/L), cos(2 pi k (x - xMin)/L)]_{k=1..K}

    Every feature is L-periodic in x together with all of its derivatives, so
    any smooth function of them is exactly L-periodic. Feeding the network these
    instead of raw x makes the periodic BC hold by construction in u, u_x, u_xx,
    ... -- which is what this third-order equation actually needs, where
    `periodic_bc` only ever penalised the mismatch in u.

    K does not band-limit the network: the hidden nonlinearities generate higher
    harmonics from these features.
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


class PINNModel(nn.Module):
    """
    Keras-equivalent MLP (glorot_uniform kernels, zero biases, 2*sigmoid-1
    activations, linear output), wrapped so that the initial and boundary
    conditions hold by construction:

        u_theta(x, t) = u_0(x) + phi(t) * N_theta(gamma(x), t),
        phi(t) = 1 - exp(-t / tau),   phi(0) = 0

    so u_theta(x, 0) = u_0(x) exactly, and gamma (PeriodicFeatures) makes
    u_theta periodic in x exactly. Neither condition needs a penalty or a dual.

    This is also what retires the trivial-solution trap documented above. u = 0
    is an exact solution of the PDE *and* a critical point of H (dH/du =
    alpha u^2 + rho u + nu u_xx vanishes there), so the soft formulation parked
    on it with a constraint gradient dying to second order -- the entire reason
    the H constraint needed a grace period. Under this ansatz u = const is not
    representable, since u(.,0) = u_0 is fixed and non-constant. What stays
    reachable is N_theta = 0, i.e. the frozen field u(x,t) = u_0(x): exactly
    feasible for every invariant constraint, but carrying a large PDE residual,
    so the objective pushes away from it rather than toward it.

    phi'(0) = 1/tau fixes the scale the network must produce to realise the
    initial time derivative: N_theta(x,0) = tau * u_t(x,0). phi is bounded on
    [0,1), unlike phi(t) = t, which would scale the output by up to tMax.

    `forward` keeps the single (N,2) input of the original so every call site is
    unchanged; x and t are split out internally. Autograd still reaches the leaf
    x and t through the cat that built xt.

    Full treatment in FORMULATION.md section 6: the lift, exactness, completeness
    (phi must have a SIMPLE zero at 0, or solutions with u_t(.,0) != 0 are
    excluded), the choice of tau and K, and the trap analysis.

    Set hard_ic / hard_bc False to recover the plain (x,t) -> u network.
    """

    def __init__(self, num_hidden_layers=depth, num_neurons_per_layer=width,
                 hard_ic=None, hard_bc=None, n_modes=None, tau=None):
        super().__init__()
        self.hard_ic = HARD_IC if hard_ic is None else hard_ic
        self.hard_bc = HARD_BC if hard_bc is None else hard_bc
        self.tau = IC_TAU if tau is None else tau
        n_modes = N_MODES if n_modes is None else n_modes

        if self.hard_bc:
            self.feat = PeriodicFeatures(n_modes, xMin, xMax - xMin)
            in_dim = 2 * n_modes + 1
        else:
            self.feat = None
            in_dim = 2

        layers = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, num_neurons_per_layer))
            layers.append(SigmoidCentered())
            in_dim = num_neurons_per_layer
        layers.append(nn.Linear(in_dim, 1))  # linear output
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        z = torch.cat([self.feat(x), t], dim=1) if self.hard_bc else xt
        out = self.net(z)
        if self.hard_ic:
            out = u_0(x) + (1.0 - torch.exp(-t / self.tau)) * out
        return out


# ── Losses ────────────────────────────────────────────────────────────────────

def periodic_bc(model, x, t):
    """Mismatch of u between the two boundaries, at every collocation time."""
    xL = torch.full_like(x, xMin)
    xR = torch.full_like(x, xMax)
    uL = model(torch.cat([xL, t], dim=1))
    uR = model(torch.cat([xR, t], dim=1))
    return torch.mean((uL - uR)**2)


def invariants_of(model, attached):
    """
    I_i(t_j) for every active invariant, on the DEDICATED quadrature grid.

    Deliberately *not* the collocation grid. The two answer different questions
    and have unrelated resolution requirements:

      * the collocation grid is sized from the parameter count (`fac`) -- an
        optimisation-theoretic choice about how many residual equations to impose;
      * the quadrature grid has to resolve the invariant *density*, which is
        quadratic and cubic in (u, u_x) and so spreads to far higher harmonics
        than u itself once solitons form.

    Tying them together was silently fatal. On the Nx=45 grid this script's own
    quadrature reports 109% drift in the Hamiltonian for the *exact* solution;
    all of it is aliasing. And the aliasing is time-dependent (identically zero
    at t=0), so the one way to make it cancel in I(t) - I(0) is for u not to
    depend on t -- handing the optimiser the frozen field u = u_0 as its only
    feasible point, which is where the CH runs went. See `quadrature_floor`.

    `attached=False` is only right when the invariants are merely monitored;
    when they drive a constraint they must stay on the graph.
    """
    xq = quad_inputs[:, 0:1].detach().requires_grad_(True)
    tq = quad_inputs[:, 1:2].detach()
    u = model(torch.cat([xq, tq], dim=1))
    u_x = torch.autograd.grad(u, xq, torch.ones_like(u), create_graph=attached)[0]
    if not attached:
        u, u_x = u.detach(), u_x.detach()
    ug, uxg = u.reshape(Nt, NX_QUAD), u_x.reshape(Nt, NX_QUAD)
    return {n: quad_boole(_DENSITIES[n](ug, uxg), dx_quad) for n in ACTIVE_INVARIANTS}


def custom_loss(inputs, model, epoch, no_grad_H=True):
    """
    Returns (loss, loss_type, pde_loss_H1, ic_loss, bc_loss, H_val).

    `loss` is the TF script's linear aggregation of the three components. In
    constrained mode the caller ignores it and builds its own objective from the
    components instead.

    `no_grad_H` detaches the Hamiltonian, which is right when H is only being
    monitored. It must be False when H feeds a constraint, otherwise no gradient
    reaches the network through the H term.
    """
    x = inputs[:, 0:1].detach().requires_grad_(True)
    t = inputs[:, 1:2].detach().requires_grad_(True)

    u_model = model(torch.cat([x, t], dim=1))

    # The TF version uses three nested persistent GradientTapes; tape.gradient of
    # a vector output is the sum-gradient, i.e. grad_outputs=ones here.
    u_x, u_t = torch.autograd.grad(
        u_model, [x, t], torch.ones_like(u_model), create_graph=True)
    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(
        u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]

    u_squared_x = 2 * u_model * u_x
    r = u_t - alpha * u_squared_x - rho * u_x - nu * u_xxx

    # === PDE residual loss (stabilised, consistent) ===
    pde_loss_L2 = torch.mean(torch.square(r))

    r_grid = r.reshape(Nt, Nx)
    L = xMax - xMin

    pde_loss_grad = torch.mean(grad_L2_fft_batch(r_grid, L))

    # mesh-scaled stabilization parameter
    lam = 0.01 * (dx**2) * min(1.0, float(epoch) / 1000.0)

    pde_loss_H1 = pde_loss_L2 + lam * pde_loss_grad

    # === Initial condition ===
    ic_mask = (torch.abs(t) < 1e-6).squeeze(-1)
    x_ic = x[ic_mask]
    t_ic = torch.zeros_like(x_ic)
    u_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))
    data_fitting_loss_0 = torch.mean(torch.square(u_ic_pred - u_0(x_ic)))

    # === Periodic BC ===
    data_fitting_loss_l_r = periodic_bc(model, x, t)

    # === Aggregation ===
    # loss, loss_type = chebyshev_loss_function(...)
    # loss, loss_type = augmented_chebyshev_loss_function(...)
    loss, loss_type = linear_loss_function(
        [pde_loss_H1, data_fitting_loss_0, data_fitting_loss_l_r],
        lambdas,
    )

    # === Conserved quantities I_i(t), one value per time slice ===
    #
    # Computed on their own, finer x-grid rather than from `u_model`/`u_x`
    # above. See `invariants_of`.
    inv_vals = invariants_of(model, attached=not no_grad_H)

    return (loss, loss_type, pde_loss_H1, data_fitting_loss_0,
            data_fitting_loss_l_r, inv_vals)


# ── Invariant quadrature grid (decoupled from the collocation grid) ──────────
#
# See `invariants_of` for why these must not be the same grid. Measured by
# pushing the EXACT reference solution through this script's own quadrature
# (type_pde=1); `*` marks a floor above the tolerance being demanded:
#
#   NX_QUAD     45         181        361       721
#   mass    3.38e-02*   3.22e-05   7.46e-09   1.9e-16
#   mom     2.12e-01*   8.04e-04*  7.40e-07   6.7e-12
#   ham     1.09e+00*   1.09e-02*  4.07e-05   1.6e-11
#
# At Nx=45 the quadrature reports 109% "drift" in the Hamiltonian for the exact
# solution -- entirely aliasing. The mom density holds harmonics up to ~99
# against a collocation Nyquist of 22, which is `NOTES.md` §5.2 arriving from a
# second direction. NX_QUAD=361 clears every floor by >100x; 721 reaches solver
# precision, if that is ever wanted.
NX_QUAD = 361
NX_QUAD += (-(NX_QUAD - 1)) % 4   # Boole needs (NX_QUAD - 1) % 4 == 0, as for Nx
QUAD_FLOOR_CHECK = True           # abort if a tolerance floor is below the quadrature floor


# ── Collocation grid ──────────────────────────────────────────────────────────
#
# np.meshgrid(x, t) with the default 'xy' indexing gives arrays of shape
# (Nt, Nx) with x_grid[i, j] = x[j] and t_grid[i, j] = t[i]. Row-major flatten
# then puts collocation point (i, j) at index i*Nx + j, so `reshape(Nt, Nx)`
# recovers one time slice per row -- which is what `H` and `r_grid` assume.

x_np = np.linspace(xMin, xMax, Nx).astype(np.float32)
t_np = np.linspace(0., tMax, Nt).astype(np.float32)
x_grid_np, t_grid_np = np.meshgrid(x_np, t_np)

inputs = torch.tensor(
    np.column_stack([x_grid_np.ravel(), t_grid_np.ravel()]),
    dtype=DTYPE, device=device,
)
x_grid = torch.tensor(x_grid_np, dtype=DTYPE, device=device)

# Invariant quadrature grid: a finer x-grid, at the same collocation times.
x_quad_np = np.linspace(xMin, xMax, NX_QUAD).astype(np.float32)
dx_quad = (xMax - xMin) / (NX_QUAD - 1)
_xq_np, _tq_np = np.meshgrid(x_quad_np, t_np)
quad_inputs = torch.tensor(
    np.column_stack([_xq_np.ravel(), _tq_np.ravel()]),
    dtype=DTYPE, device=device,
)

# The reference invariant values I^0 are computed further down, with the 
# invariant configuration (see REF).


# ── Training setup ────────────────────────────────────────────────────────────

model = PINNModel().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters : {n_params} (Ntheta from Nx_from_arch: {Ntheta})")
print(f"Grid       : Nx={Nx}, Nt={Nt}, collocation points={Nx * Nt}")
print(f"Spacing    : dx={dx:.6f}, dt={dt:.6f}, h={h:.6f}")
print(f"Quadrature : NX_QUAD={NX_QUAD} (dx={dx_quad:.6f}) -> {NX_QUAD * Nt} points "
      f"for the invariants, vs {Nx * Nt} for the residual")

epochs = 2000

learning_rate = 1e-4
learning_rate_type = 'cosineDecay'


def keras_cosine_decay(step, decay_steps=1000, alpha=0.5):
    """
    Multiplier reproducing keras.optimizers.schedules.CosineDecay.

    With warmup_target=None Keras ignores warmup_steps entirely and clamps the
    step to decay_steps, so the rate decays from 1.0 to alpha over decay_steps
    and stays at alpha afterwards.
    """
    s = min(float(step), float(decay_steps))
    cosine_decayed = 0.5 * (1.0 + np.cos(np.pi * s / decay_steps))
    return (1.0 - alpha) * cosine_decayed + alpha


def make_optimizer(constrained):
    """
    Unconstrained: the TF setting exactly (AdamW + Keras CosineDecay).

    Constrained: the TF learning rate is far too small here. Under constraints
    the objective is the PDE residual alone and the data-fitting terms enter
    only through the Lagrangian, so the net first finds the trivial near-zero
    solution (which satisfies the PDE) and then has to climb out of it. At
    lr=1e-4 decaying to 5e-5 the IC term moves by ~0.02 per 400 epochs, i.e. it
    never gets there. Use the Camassa-Holm script's setting instead.
    """
    if constrained or True:
        opt = torch.optim.NAdam(model.parameters(), lr=5e-2)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9**(1/epochs))
        return opt, sch, 1e-2, 'expDecay'

    # keras.optimizers.AdamW defaults to weight_decay=0.004 (torch defaults to
    # 0.01); both apply it decoupled as p -= lr*wd*p, so only the value differs.
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                            betas=(0.8, 0.9), eps=1e-07, weight_decay=0.004)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=keras_cosine_decay)
    return opt, sch, learning_rate, learning_rate_type

# Plot the solution every PLOT_EVERY epochs during training (set to None to
# disable). Deviation from the TF script, which only plots at the end.
PLOT_EVERY = None


# ── Constrained formulation (dual_optim) ──────────────────────────────────────
#
# The initial and periodic boundary conditions hold by construction (see
# PINNModel), so what is left is to minimise the PDE residual subject to
# conservation of the invariant hierarchy:
#
#     min_theta  ||r(theta)||_H1^2
#     s.t.       |I_i(t_j; theta) - I_i^0|  <=  eps_i * S_i
#                    for each active invariant i and each collocation time t_j
#                    (or one constraint on the max over j, if MAX_H)
#
# with S_i the tolerance scale (see EPS_SCALE). Set HARD_IC / HARD_BC False to
# put the IC / BC back as constrained groups instead (the previous formulation),
# and CONSTRAINED = False to recover the faithful translation of the TF script
# (plain weighted-sum loss, invariants monitored only).
#
# dual_optim expects constraints in the form c(theta) <= 0, so every entry built
# below is "value - tolerance".

MAX_H = True          # collapse each invariant's Nt constraints into their max

# A condition that holds by construction needs neither a penalty nor a dual.
CONSTRAIN_IC = not HARD_IC
CONSTRAIN_BC = not HARD_BC

# ── Invariants to constrain ───────────────────────────────────────────────────
#
# Selects from the KdV hierarchy, in the order the dual groups are created.
# C0 is linear in u, C1 quadratic, H cubic-plus-gradient; the low-order members
# cost one quadrature each and are far better conditioned than H.
ACTIVE_INVARIANTS = (
    'mass',
    'mom',
    # 'ham'
)

_DENSITIES = {'mass': kdv_mass, 'mom': kdv_mom, 'ham': kdv_density}
_EPS_FLOOR = {'mass': 1e-2, 'mom': 1e-1, 'ham': 1e-1}
_DUAL_LR   = {'mass': 1e-1, 'mom': 1e-1, 'ham': 1e-1}

# Tolerance scale S_i. 'l1' uses int|density|dx, 'value' uses |I^0|.
#
# 'value' is the conventional "relative error in the invariant", but it is
# degenerate here: for type_pde=1, int cos(pi x) dx = 0 exactly over [0,2], so
# the mass scale would be zero and the constraint would silently demand exact
# conservation. |I^0| is also tiny for the Hamiltonian (2.39e-3, pure
# dispersion, since int cos^3 = 0 kills the cubic part) while the density it
# integrates is O(0.14) -- so scaling by |I^0| asks for the *cancellation* to be
# preserved, not just the drift to be small. 'l1' never vanishes for a
# nontrivial field and coincides with |I^0| whenever the density has one sign.
# Both numbers are reported: drift/S drives the constraint, drift/|I^0| is
# logged alongside it for comparability with the literature.
EPS_SCALE = 'l1'

# With the IC hard-encoded the trivial-solution trap is gone (see PINNModel):
# u = const is no longer representable, so the invariant constraints do not need
# to be held back. Raise this when running with HARD_IC = False, where u = 0 is
# both an exact PDE solution and a critical point of H and the net will park on
# it with a constraint gradient that dies to second order.
GRACE = 1000 if HARD_IC else 1000
IC_EPS = 1e-6
BC_EPS = 1e-6
ICBC_DUAL_LR = 1e-1

# ── Spectral reference solution (NOTES.md §5.8) ───────────────────────────────
#
# Every other diagnostic here is self-referential: the residual and the
# invariant drift are both computed from the network's own output, so a wrong
# field can score well on them. The frozen field u(x,t) = u_0(x) conserves every
# invariant *exactly* and is completely wrong; nothing above can tell.
#
# KdV is integrated with an integrating-factor RK4. In Fourier space
#
#     u_t = alpha (u^2)_x + rho u_x + nu u_xxx    ==>
#     v_t = alpha (ik) F[u^2] + L(k) v,    L(k) = i (rho k - nu k^3),
#
# and the substitution w = e^{-L tau} v removes the linear part exactly,
#
#     w_tau = e^{-L tau} alpha (ik) F[u^2].
#
# This matters: u_xxx is what makes an explicit scheme stiff. Plain RK4 needs
# dt <~ 2.8/(|nu| k_max^3), which is ~4.5e5 steps for type_pde=1; solving the
# linear part exactly leaves only the nonlinear CFL, three orders of magnitude
# gentler. tau is measured from the start of each step, so the phase e^{L tau}
# never grows large enough to lose accuracy.
#
# The solver checks itself against the same three invariants the constraints
# use: a reference that cannot conserve them is not fit to be a baseline, and
# the run aborts rather than reporting errors against it. No dealiasing is
# applied -- for a resolved solution the top modes are negligible, and if they
# are not, the self-check is what says so.

REF_ON        = True    # compute the reference and report true L2 errors
REF_NX        = 512     # spectral modes; periodic grid, right endpoint excluded
REF_NT        = 101     # time slices at which the error is evaluated
REF_DT_SAFETY = 8.0     # substep so that h * (|nu| k_max^3 + c k_max) <= this
REF_EVERY     = 25      # evaluate the network against the reference every N epochs
REF_MAX_DRIFT = 1e-6    # abort if the reference's own invariant drift exceeds this


def reference_solution():
    """Integrating-factor RK4 reference for KdV. Returns (x, t, U), U of shape (Nx, Nt)."""
    L_dom = xMax - xMin
    xr = xMin + L_dom * np.arange(REF_NX) / REF_NX   # periodic grid: excludes xMax
    tr = np.linspace(0.0, tMax, REF_NT)
    dx_r = L_dom / REF_NX
    k = 2.0 * np.pi * np.fft.fftfreq(REF_NX, d=dx_r)
    Lk = 1j * (rho * k - nu * k ** 3)                # the linear part, solved exactly

    u0 = u_0(torch.as_tensor(xr, dtype=torch.float64)).numpy()

    def g(tau, w):
        """w_tau, i.e. the nonlinear term pulled back through the integrating factor."""
        u = np.real(np.fft.ifft(np.exp(Lk * tau) * w))
        return np.exp(-Lk * tau) * (alpha * 1j * k * np.fft.fft(u * u))

    # Timestep. Two limits apply -- the nonlinear CFL from the characteristic
    # speed of alpha(u^2)_x + rho u_x, and a dispersive one from nu u_xxx -- and
    # for KdV it is the dispersive term that binds, by three orders of
    # magnitude. Measured at N=512, t_max=5: h*(|nu|k^3 + c k) = 62 gives a
    # Hamiltonian drift of 3.6e-01, = 16 gives 9.9e-11, and = 4 gives 3.0e-12,
    # while refining the *grid* to N=2048 at the coarse step does not help at
    # all (and N=1024 there overflows outright). The integrating factor removes
    # the linear stability limit but not the accuracy cost of a large |L(k)|h
    # across the RK4 stages, so the step is set from the stiff rate.
    speed = 2.0 * abs(alpha) * np.abs(u0).max() + abs(rho)
    kmax = np.abs(k).max()
    rate = abs(nu) * kmax ** 3 + max(speed, 1e-12) * kmax
    dt_slice = tr[1] - tr[0]
    sub = max(1, int(np.ceil(dt_slice * rate / REF_DT_SAFETY)))
    h = dt_slice / sub
    E = np.exp(Lk * h)

    U = np.empty((REF_NX, REF_NT))
    U[:, 0] = u0
    v = np.fft.fft(u0)
    for j in range(1, REF_NT):
        for _ in range(sub):
            a = g(0.0, v); b = g(h / 2, v + h * a / 2)
            c = g(h / 2, v + h * b / 2); d = g(h, v + h * c)
            v = E * (v + h * (a + 2 * b + 2 * c + d) / 6)
        U[:, j] = np.real(np.fft.ifft(v))

    # Self-check, against the script's own density functions so the two cannot
    # disagree. Scaled by the L1 norm of the density, not |I^0|: for type_pde=1
    # the mass int cos(pi x) dx vanishes exactly, and dividing by it would turn
    # a perfectly conserved invariant into a spurious failure.
    def D(f, n=1):
        return np.real(np.fft.ifft((1j * k) ** n * np.fft.fft(f)))

    Ux = np.stack([D(U[:, j]) for j in range(REF_NT)], axis=1)
    drift = {}
    for n, f in _DENSITIES.items():
        I = np.array([f(torch.as_tensor(U[:, j]), torch.as_tensor(Ux[:, j])).numpy().sum() * dx_r
                      for j in range(REF_NT)])
        S = np.abs(f(torch.as_tensor(U[:, 0]), torch.as_tensor(Ux[:, 0])).numpy()).sum() * dx_r
        drift[n] = np.abs(I - I[0]).max() / (S + 1e-30)

    print(f"Reference  : integrating-factor RK4, N={REF_NX}, {REF_NT} slices, "
          f"{sub} substeps/slice (dt={h:.2e}, h*rate={h * rate:.1f})")
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
        _REF_GRID = torch.tensor(
            np.column_stack([Xg.ravel(), Tg.ravel()]), device=device, dtype=DTYPE)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        Up = model(_REF_GRID).cpu().numpy().reshape(U.shape)
    if was_training:
        model.train()
    err = Up - U
    per_slice = np.linalg.norm(err, axis=0) / (np.linalg.norm(U, axis=0) + 1e-30)
    return np.linalg.norm(err) / np.linalg.norm(U), per_slice, err


REFSOL = reference_solution() if REF_ON else None
if REFSOL is not None:
    # The score of the frozen field u(x,t) = u_0(x). It conserves every
    # invariant exactly, so it is the number a run lands on when the constraints
    # are satisfied and no dynamics have been learned -- a tripwire, not a target.
    REF_FROZEN = (np.linalg.norm(REFSOL[2] - REFSOL[2][:, :1])
                  / np.linalg.norm(REFSOL[2]))
    print(f"             frozen field u=u_0 scores rel. L2 = {REF_FROZEN:.4f}")
else:
    REF_FROZEN = float('nan')


# ── Dual group layout ─────────────────────────────────────────────────────────
#
# One entry per dual group, in registration order. `build_constraints` walks
# this same list, so registration and assembly cannot drift apart -- the failure
# mode the earlier hand-written offsets had.
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

# Reference value I^0 and tolerance scale S for each invariant, from the exact
# initial condition and the same quadrature the constraint uses.
# On the QUADRATURE grid, with the quadrature spacing, so the systematic rule
# error still cancels exactly in I(t) - I^0 at t = 0.
_x1d = torch.tensor(x_quad_np, dtype=DTYPE, device=device)
_u0, _u0x = u_0(_x1d), u_0_x(_x1d)
REF = {}
for _n in ACTIVE_INVARIANTS:
    _f0 = _DENSITIES[_n](_u0, _u0x)
    _I0 = quad_boole(_f0, dx_quad)
    _L1 = quad_boole(_f0.abs(), dx_quad)
    _S = _L1 if EPS_SCALE == 'l1' else torch.abs(_I0)
    # The degeneracy test has to be RELATIVE to the density's L1 norm. An
    # exactly-zero integral does not come back as zero in float32 -- the KdV
    # mass evaluates to |I0| = 3.3e-8, not 0 -- so an `<= 0` test passes it
    # through and the constraint then silently demands conservation to 1e-8.
    # |I0|/L1 is 2.6e-8 there against 1.7e-2 for the Hamiltonian, so the two
    # are cleanly separated.
    if _S.item() <= 1e-6 * _L1.item():
        raise SystemExit(
            f"invariant {_n!r}: tolerance scale {_S.item():.3e} is degenerate "
            f"against the density L1 norm {_L1.item():.3e} "
            f"(EPS_SCALE={EPS_SCALE!r}) -- use EPS_SCALE='l1'")
    REF[_n] = (_I0, _S)
    print(f"Invariant {_n:<5}: I0={_I0.item():+.6e}  scale={_S.item():.6e}")

# ── Quadrature floor: is the tolerance we demand even measurable? ────────────
#
# The failure this guards against is not hypothetical -- it is what made the
# constrained CH runs converge to the frozen field. See `invariants_of`.
#
# Push the EXACT reference solution through this script's own quadrature and ask
# what drift it reports. The true answer is ~1e-12 (the reference conserves to
# solver precision, see `reference_solution`), so whatever comes back is pure
# quadrature error. Any tolerance below it is unmeasurable, and because the
# error is time-dependent while a t-independent field's cancels exactly,
# demanding it hands the optimiser the frozen field as its only feasible point.

def quadrature_floor(refsol):
    """Drift this script's quadrature reports for the exact reference solution."""
    xr, tr, Ur = refsol
    N, L_dom = len(xr), xMax - xMin
    kk = 2.0 * np.pi * np.fft.fftfreq(N, d=L_dom / N)
    Uxr = np.real(np.fft.ifft((1j * kk)[:, None] * np.fft.fft(Ur, axis=0), axis=0))
    # exact trigonometric interpolation onto the quadrature grid
    ang = np.exp(1j * np.outer(x_quad_np.astype(np.float64) - xr[0], kk))
    Uq = np.real(ang @ (np.fft.fft(Ur, axis=0) / N))
    Uxq = np.real(ang @ (np.fft.fft(Uxr, axis=0) / N))
    out = {}
    for n in ACTIVE_INVARIANTS:
        f = _DENSITIES[n](torch.as_tensor(Uq.T), torch.as_tensor(Uxq.T))
        I = quad_boole(f, dx_quad)
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


if CONSTRAINED:
    if not GROUPS:
        raise SystemExit("CONSTRAINED=True but no constraint group is enabled")
    from humancompatible.train.dual_optim import ALM

    # penalty=0 makes this a pure Lagrangian (no quadratic augmentation term).
    dual_opt = ALM(m=GROUPS[0].m, lr=GROUPS[0].lr, penalty=0., is_ineq=True,
                   momentum=0., dual_range=(0, 10000), device=device)
    for g in GROUPS[1:]:
        dual_opt.add_constraint_group(m=g.m, lr=g.lr, is_ineq=True)
    print("Dual groups: " + ", ".join(f"{g.name}(m={g.m}, lr={g.lr:g})"
                                      for g in GROUPS))

# Check the hard-encoded conditions on the untrained network: with the ansatz in
# place both residuals should sit at float32 round-off, not merely be small.
with torch.no_grad():
    _xc = torch.tensor(np.linspace(xMin, xMax, 501, dtype=np.float32),
                       dtype=DTYPE, device=device).reshape(-1, 1)
    _tc = torch.rand(501, 1, dtype=DTYPE, device=device) * tMax
    _ic_res = (model(torch.cat([_xc, torch.zeros_like(_xc)], dim=1))
               - u_0(_xc)).abs().max().item()
    _bc_res = (model(torch.cat([torch.full_like(_tc, xMin), _tc], dim=1))
               - model(torch.cat([torch.full_like(_tc, xMax), _tc], dim=1))
               ).abs().max().item()
# Reported relative to ||u_0||_inf: under the hard ansatz phi(0) is exactly 0,
# so the only residual left is a 1-ulp difference from evaluating u_0 on the
# strided view xt[:, 0:1] rather than on a contiguous tensor. Anything above a
# few float32 eps means the ansatz is not doing what it claims.
_u0_scale = u_0(_xc).abs().max().item() + 1e-30
print(f"Check      : max|u(x,0)-u_0(x)|={_ic_res:.3e} "
      f"(rel {_ic_res/_u0_scale:.1e}){'  [hard]' if HARD_IC else ''},  "
      f"max|u(xMin,t)-u(xMax,t)|={_bc_res:.3e}"
      f"{'  [hard]' if HARD_BC else ''}")

# Primal optimizer depends on the formulation; see make_optimizer.
optimizer, scheduler, learning_rate, learning_rate_type = make_optimizer(CONSTRAINED)
print(f"Optimizer  : {type(optimizer).__name__}(lr={learning_rate}), "
      f"schedule={learning_rate_type}, constrained={CONSTRAINED}")


def eps_of(g, epoch):
    """
    Relative tolerance for group `g` at `epoch`.

    Invariant groups follow the continuation schedule eps = max(1/(epoch-GRACE),
    floor): loose when the constraint switches on, tightening toward `floor`.
    Condition groups (IC / BC, present only when not hard-encoded) carry a fixed
    absolute tolerance instead.
    """
    if g.kind != 'inv':
        return g.eps_floor
    if epoch <= GRACE:
        return float('inf')          # inactive
    return max(1.0 / (epoch - GRACE), g.eps_floor)


def build_constraints(inv_vals, ic_loss, bc_loss, epoch):
    """
    One constraint tensor per dual group, in GROUPS order, plus a dict of raw
    values for logging: report[name] = (drift/S, drift/|I^0|).

    Invariant groups impose, for each collocation time t_j,

        c_i(t_j) = |I_i(t_j) - I_i^0| - eps_i * S_i  <=  0,

    a bound on the ABSOLUTE drift whose tolerance carries the scale S_i. Same
    feasible set as bounding drift/S_i, but the gradient is not amplified by
    1/S_i. Do NOT also divide the drift by S_i -- the tolerance already carries
    it (that double-application was a real bug here once).

    Blocks keep their full width (1 under MAX_H, else Nt) even while a group is
    inactive during the grace period, so the list arity is invariant across all
    flag combinations; zeros leave the dual update and the Lagrangian untouched.
    """
    blocks, report = [], {}

    for g in GROUPS:
        if g.kind == 'inv':
            I0 = REF[g.name][0] if USE_TRUE_H0 else inv_vals[g.name][0]
            S = REF[g.name][1]
            drift = torch.abs(inv_vals[g.name] - I0)          # (Nt,)
            aI0 = torch.abs(I0)
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
    hard-encoded nor constrained (folded in with its TF weight). With HARD_IC
    and HARD_BC this is the PDE residual alone --

        min_theta  ||r(theta)||_{H1}^2   s.t.  the invariant drift constraints.
    """
    terms, weights = [pde_loss], [lambdas[0]]
    if not (HARD_IC or CONSTRAIN_IC):
        terms.append(ic_loss); weights.append(lambdas[1])
    if not (HARD_BC or CONSTRAIN_BC):
        terms.append(bc_loss); weights.append(lambdas[2])
    if len(terms) == 1:
        return pde_loss
    return linear_loss_function(terms, torch.stack(weights))[0]


# ── Solution plotting (surface + IC / BC traces) ──────────────────────────────

def _eval_surface(model, N=600):
    """u_theta on a square (N+1)^2 grid. Returns U indexed as U[x_index, t_index]."""
    tspace = np.linspace(0.0, tMax, N + 1, dtype=np.float32)
    xspace = np.linspace(xMin, xMax, N + 1, dtype=np.float32)
    T_plot, X_plot = np.meshgrid(tspace, xspace)
    XT = torch.tensor(
        np.column_stack([X_plot.ravel(), T_plot.ravel()]), dtype=DTYPE, device=device
    )
    with torch.no_grad():
        u = model(XT)
    return xspace, tspace, X_plot, T_plot, u.cpu().numpy().reshape(N + 1, N + 1)


def _boundary_u_x(model, tspace):
    """u_x at x=xMin and x=xMax, to check periodicity of the derivative."""
    t = torch.tensor(tspace, dtype=DTYPE, device=device).reshape(-1, 1)
    out = []
    for xb in (xMin, xMax):
        x = torch.full_like(t, xb).requires_grad_(True)
        u = model(torch.cat([x, t], dim=1))
        u_x = torch.autograd.grad(u, x, torch.ones_like(u))[0]
        out.append(u_x.detach().cpu().numpy().ravel())
    return out


def plot_solution(model, tag='', surf_path=None, edge_path=None):
    """
    Two figures:
      1. 3-D surface with the IC (exact vs learned) and both boundary traces on it
      2. 1-D panels: IC fit, BC value periodicity, BC derivative periodicity

    NOTE: `periodic_bc` only penalises the mismatch in u, not in u_x, so the
    third panel is a pure diagnostic -- nothing in the loss drives it to zero.
    """
    xspace, tspace, X_plot, T_plot, U = _eval_surface(model)
    u0_exact = u_0(torch.tensor(xspace, dtype=DTYPE, device=device)).cpu().numpy()
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
    ax.set_title(f'Solution to KdV equation{tag}')
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
    # plt.show()
    # plt.close(fig)


# ── Training loop ─────────────────────────────────────────────────────────────

# H_* keys track one designated invariant so the existing plots and CSV columns
# keep their meaning; the Hamiltonian if active, else the first invariant group.
H_KEY = 'ham' if any(g.name == 'ham' for g in GROUPS) else (
    GROUPS[0].name if GROUPS and GROUPS[0].kind == 'inv' else None)

hist = {k: [] for k in ['loss', 'pde', 'ic', 'bc', 'lagrangian', 'ref_l2',
                        'H_min', 'H_max', 'H_mean', 'H_std',
                        'H_abs_err', 'H_rel_err']}
ref_l2_last = float('nan')
for g in GROUPS:
    hist[f'drift_{g.name}'] = []
    hist[f'reldrift_{g.name}'] = []
    hist[f'dual_{g.name}'] = []

lambdas_values = [lambdas.cpu().numpy()]
cheb_par_values = [cheb_par.cpu().numpy()]

t0 = time()
for epoch in range(epochs):
    if PLOT_EVERY is not None and (epoch % PLOT_EVERY == 0 or epoch == epochs - 1):
        model.eval()
        plot_solution(model, tag=f' (epoch {epoch})')

    model.train()
    optimizer.zero_grad()

    # The invariants must stay attached to the graph when they drive constraints.
    loss, loss_type, pde_loss, ic_loss, bc_loss, inv_vals = custom_loss(
        inputs, model, epoch, no_grad_H=not CONSTRAINED)

    if CONSTRAINED:
        loss = build_objective(pde_loss, ic_loss, bc_loss)
        blocks, report = build_constraints(inv_vals, ic_loss, bc_loss, epoch)
        # Passing the per-group list (not a flat tensor) makes dual_optim verify
        # each group's arity, so a layout slip raises instead of silently
        # feeding one group's values to another's duals.
        lagrangian = dual_opt.forward_update(loss, blocks)
        lagrangian.backward()
    else:
        lagrangian = loss
        _, report = build_constraints(inv_vals, ic_loss, bc_loss, epoch)
        loss.backward()

    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        hist['loss'].append(loss.item())
        hist['pde'].append(pde_loss.item())
        hist['ic'].append(ic_loss.item())
        hist['bc'].append(bc_loss.item())
        hist['lagrangian'].append(lagrangian.item())

        # True error against the reference, every REF_EVERY epochs rather than
        # every one; nan elsewhere so the column stays one row per epoch.
        if REFSOL is not None and (epoch % REF_EVERY == 0 or epoch == epochs - 1):
            ref_l2_last = ref_errors(model, REFSOL)[0]
            hist['ref_l2'].append(ref_l2_last)
        else:
            hist['ref_l2'].append(float('nan'))

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
            # tf.math.reduce_std is the population std, so correction=0.
            hist['H_std'].append(H_val.std(correction=0).item())
            H_abs_error = torch.abs(H_val - I0)
            hist['H_abs_err'].append(H_abs_error.max().item())
            # The TF script reports the *last* time slice here, not the max.
            hist['H_rel_err'].append(
                (H_abs_error / torch.abs(I0 + 1e-16))[-1].item())

    if len(hist['loss']) > 1 and do_training:
        # SoftAdapt weight update (inactive in the original: do_training = False)
        num1 = np.exp(hist['pde'][-1] - hist['pde'][-2])
        num2 = np.exp(hist['ic'][-1] - hist['ic'][-2])
        num3 = np.exp(hist['bc'][-1] - hist['bc'][-2])
        den = num1 + num2 + num3
        lambdas = torch.tensor([num1 / den, num2 / den, num3 / den],
                               dtype=DTYPE, device=device)
        lambdas_values.append(lambdas.cpu().numpy())

    if epoch % 100 == 0 or epoch == epochs - 1:
        parts = ' '.join(
            f"{g.name}={hist[f'drift_{g.name}'][-1]:.2e}"
            f"(mu={hist[f'dual_{g.name}'][-1]:.2e})" for g in GROUPS)
        print(f"Epoch {epoch + 1:5d}/{epochs} | obj={loss.item():.4e} "
              f"L={hist['lagrangian'][-1]:.4e} | "
              f"IC={hist['ic'][-1]:.2e} BC={hist['bc'][-1]:.2e} | {parts} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}"
              + (f" | relL2={ref_l2_last:.3e}" if REFSOL is not None else ""))

if CONSTRAINED and GROUPS:
    print("\nFinal constraint status (drift/scale vs tolerance, <=0 satisfied):")
    for g in GROUPS:
        eps = eps_of(g, epochs - 1)
        d = hist[f'drift_{g.name}'][-1]
        extra = ('' if g.kind != 'inv'
                 else f"   [|dI|/|I0| = {hist[f'reldrift_{g.name}'][-1]:.3e}]")
        print(f"  {g.name:<5}: {d:.4e} vs {eps:.4e}  -> {d - eps:+.3e}"
              f"   (mu={hist[f'dual_{g.name}'][-1]:.3e}){extra}")

print(f"\nLoss type                    : {loss_type}")
print(f"IC MSE (diagnostic)          : {hist['ic'][-1]:.4e}"
      f"{'  [hard-encoded]' if HARD_IC else ''}")
print(f"BC MSE (diagnostic)          : {hist['bc'][-1]:.4e}"
      f"{'  [hard-encoded]' if HARD_BC else ''}")
if H_KEY is not None:
    print(f"Hamiltonian mean             : {hist['H_mean'][-1]}")
    print(f"Hamiltonian standard dev.    : {hist['H_std'][-1]}")
    print(f"Hamiltonian maximum          : {hist['H_max'][-1]}")
    print(f"Hamiltonian minimum          : {hist['H_min'][-1]}")
    print(f"Hamiltonian relative error   : {hist['H_rel_err'][-1]}")
if REFSOL is not None:
    REF_L2, REF_PER_SLICE, REF_ERR = ref_errors(model, REFSOL)
    print(f"Rel. L2 vs reference         : {REF_L2:.4e}")
    print(f"  frozen field u=u_0         : {REF_FROZEN:.4e}"
          f"   -> {'NO dynamics learned' if REF_L2 > 0.95 * REF_FROZEN else 'better than frozen'}")
    print(f"  worst time slice           : {REF_PER_SLICE.max():.4e}"
          f" at t={REFSOL[1][REF_PER_SLICE.argmax()]:.2f}")
print(f"\nComputation time: {time() - t0} seconds")


# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_save_fig_string(tag, epochs, learning_rate_type, loss_type):
    return f"{RESULTS_DIR}/{tag}_epochs_{epochs}_lr_{learning_rate_type}_{loss_type}.png"


def save_show(tag):
    if save_fig:
        plt.savefig(generate_save_fig_string(tag, epochs, learning_rate_type, loss_type),
                    dpi=300)
    plt.show()
    plt.close()


# Loss history
plt.figure()
plt.semilogy(hist['loss'], label='Total Loss')
plt.semilogy(hist['pde'], label='PDE Loss')
plt.semilogy(hist['ic'], label='Initial Conditions Loss')
plt.semilogy(hist['bc'], label='Periodic Boundary Conditions Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Contributions')
plt.legend()
plt.grid()
save_show('loss')

# Hamiltonian min / max
plt.figure()
plt.plot(hist['H_min'], label='H_loss_min')
plt.plot(hist['H_max'], label='H_loss_max')
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian')
plt.title('Hamiltonian over epochs')
plt.legend()
plt.grid()
save_show('H_loss')

# Hamiltonian mean +/- std
H_mean_arr = np.array(hist['H_mean'])
H_std_arr = np.array(hist['H_std'])
plt.figure()
plt.plot(H_mean_arr)
plt.fill_between(range(len(H_mean_arr)), H_mean_arr - H_std_arr, H_mean_arr + H_std_arr,
                 alpha=0.2)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian mean')
plt.title('Hamiltonian mean over epochs with standard deviation')
plt.grid()
save_show('H_loss_mean')

# Hamiltonian std
plt.figure()
plt.plot(H_std_arr)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian std')
plt.title('Hamiltonian standard deviation over epochs')
plt.grid()
save_show('H_loss_std')

# Hamiltonian absolute error
plt.figure()
plt.plot(np.array(hist['H_abs_err']))
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian absolute error')
plt.title('Hamiltonian absolute error over epochs')
plt.grid()
save_show('H_loss_abs_error')

# Hamiltonian relative error
plt.figure()
plt.plot(np.array(hist['H_rel_err']))
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian relative error')
plt.title('Hamiltonian relative error over epochs')
plt.grid()
save_show('H_loss_rel_error')

# True L2 error against the spectral reference (NOTES.md §5.8)
if REFSOL is not None:
    _xr, _tr, _Ur = REFSOL
    _e = np.array(hist['ref_l2'])
    _k = np.isfinite(_e)
    plt.figure()
    plt.semilogy(np.arange(len(_e))[_k], _e[_k], label='rel. $L^2$ vs reference')
    plt.axhline(REF_FROZEN, ls='--', c='r', label=f'frozen $u_0$ ({REF_FROZEN:.3f})')
    plt.xlabel('Epoch'); plt.ylabel('relative $L^2$ error')
    plt.title('True error against the spectral reference')
    plt.legend(); plt.grid()
    save_show('ref_l2')

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
    _im = ax[1, 0].pcolormesh(_tr, _xr, REF_ERR, cmap='RdBu_r', vmin=-_m, vmax=_m,
                              shading='auto')
    ax[1, 0].set_xlabel('$t$'); ax[1, 0].set_ylabel('$x$')
    ax[1, 0].set_title(f'$u_\\theta - u$  (max {_m:.2e})')
    fig.colorbar(_im, ax=ax[1, 0])
    ax[1, 1].semilogy(_tr, REF_PER_SLICE, label='network')
    ax[1, 1].axhline(REF_FROZEN, ls='--', c='r', label='frozen $u_0$ (global)')
    ax[1, 1].set_xlabel('$t$'); ax[1, 1].set_ylabel('relative $L^2$ error')
    ax[1, 1].set_title(f'Error per time slice (global {REF_L2:.3e})')
    ax[1, 1].legend(); ax[1, 1].grid()
    fig.suptitle('KdV: network against the spectral reference')
    fig.tight_layout()
    if save_fig:
        fig.savefig(generate_save_fig_string('ref_compare', epochs,
                                             learning_rate_type, loss_type), dpi=300)
    plt.show()
    plt.close(fig)

# Constraint violations and their duals, one curve per group
if CONSTRAINED and GROUPS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for g in GROUPS:
        axes[0].semilogy(np.maximum(hist[f'drift_{g.name}'], 1e-16), label=g.name)
        axes[1].plot(hist[f'dual_{g.name}'], label=f'$\\mu_{{{g.name}}}$')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('drift / scale  (untolerated)')
    axes[0].set_title('Constraint values')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dual variable')
    axes[1].set_title('Dual variables')
    for a_ in axes:
        a_.legend(fontsize=8); a_.grid()
    fig.tight_layout()
    save_show('constraints')

# Solution surface, with the IC and BC traces drawn on it, plus 1-D diagnostics
model.eval()
plot_solution(
    model,
    surf_path=generate_save_fig_string('sol', epochs, learning_rate_type, loss_type),
    edge_path=generate_save_fig_string('ic_bc', epochs, learning_rate_type, loss_type),
)


# ── Save training history ─────────────────────────────────────────────────────

cols = {'epoch': list(range(1, len(hist['loss']) + 1)),
        'total_loss': hist['loss'],
        'pde_loss': hist['pde'],
        'data_fitting_loss_0': hist['ic'],
        'data_fitting_loss_l_r': hist['bc'],
        'lagrangian': hist['lagrangian']}
if H_KEY is not None:
    cols.update({'H_loss_min': hist['H_min'], 'H_loss_max': hist['H_max'],
                 'H_loss_mean': hist['H_mean'], 'H_loss_std': hist['H_std'],
                 'H_loss_abs_error': hist['H_abs_err'],
                 'H_loss_rel_error': hist['H_rel_err']})
if REFSOL is not None:
    cols['ref_rel_l2'] = hist['ref_l2']
for _g in GROUPS:
    cols[f'drift_{_g.name}'] = hist[f'drift_{_g.name}']
    cols[f'reldrift_{_g.name}'] = hist[f'reldrift_{_g.name}']
    cols[f'dual_{_g.name}'] = hist[f'dual_{_g.name}']
df = pd.DataFrame(cols)

csv_path = f'{RESULTS_DIR}/training_history_pytorch.csv'
df.to_csv(csv_path, index=False)
print(f"Saved training history to {csv_path}")
