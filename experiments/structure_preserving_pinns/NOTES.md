# structure_preserving_pinns — analysis notes

Read of the six scripts in this directory as of 2026-09-09. Numbers marked *(verified)* were
recomputed here, not taken from comments.

## 1. Inventory

Three PDE cases, each a TensorFlow original plus a PyTorch descendant.

| file | lines | state |
| --- | --- | --- |
| `CamassaHolm_trainableH1.py` | 616 | canonical, constrained, runs |
| `kdv_trainableH1.py` | 937 | verified faithful port + constrained mode, runs |
| `2D_trainableACS.py` | 291 | **does not run** (§5.4) |
| `CamassaHolm_trainableH1_tf.py` | 728 | reference only |
| `kdv_trainableH1_tf.py` | 768 | reference only |
| `2D_trainableACS_tf.py` | 758 | reference only |

**None of the three TF originals contains any constraint machinery.** They are soft-penalty PINNs
that scalarize PDE residual + IC + BC (linear / Chebyshev / smooth-Chebyshev / augmented-Chebyshev)
and *monitor* `H`. The constrained formulation is entirely the PyTorch-side contribution.

Naming: `H1` = the residual is measured in the H¹ norm (`‖r‖² + λ‖r_x‖²`), not the Hamiltonian.
`trainable` = the SoftAdapt loss weights `lambdas` (commented out in torch CH, `do_training=False`
in KdV, so inactive everywhere). `ACS` = augmented Chebyshev scalarization.

## 2. The formulation

Each PDE is Hamiltonian: `u_t = ∂_x (δH/δu)`, so `H(t) = ∫ density dx` is conserved analytically.

| case | equation | `H` density |
| --- | --- | --- |
| Camassa–Holm | `u_t - u_xxt + 3u u_x - 2u_x u_xx - u u_xxx = 0` | `u³ + u u_x²` |
| KdV | `u_t - α(u²)_x - ρu_x - ν u_xxx = 0` | `V(u) - ν u_x²/2`, `V = αu³/3 + ρu²/2` |
| 2D (ZK-like) | `u_t + u u_x + u_xxx + u_xyy = 0` | `|∇u|²/2 - u³/6` |

A soft-penalty PINN has no reason to conserve `H`. The idea here is to move the drift out of the
loss and into the constraint set:

```
min_θ  ‖r(θ)‖²_{H¹}
s.t.   |H(t_j; θ) − H0| ≤ ε_rel·|H0|      j = 1..Nt   (or one constraint on the max)
       MSE_IC ≤ 1e-6
       MSE_BC ≤ 1e-6
```

handled by `humancompatible.train.dual_optim`. With IC and BC both constrained the primal objective
reduces to the PDE residual alone.

**Framing point worth stating in any write-up.** `H(t) = H0` is a *consequence* of the PDE plus the
IC, not independent information. At the exact solution the constraint is redundant. All of its value
is in the discrete, finite-capacity regime — it is the PINN analogue of a structure-preserving
integrator, and that is the claim the experiment has to support.

## 3. Shared design (CH and KdV)

- **Grid from architecture.** `Nx = Nt = sqrt(Ntheta/fac)`, `fac=10` — deliberately *fewer*
  collocation points than parameters. width=80, depth=4 → `Ntheta=19761`, `Nx=Nt=44`, 1936
  collocation points *(verified)*.
- **Group order fixes the layout**: `[H, IC, BC]`, only enabled groups contribute a block, H block
  keeps width `m_H` even during grace so the vector length is invariant.
- **abs-form H constraint**: `c = |H − H0| − ε_rel·|H0|`. Same feasible set as bounding the relative
  drift, but the gradient is not amplified by `1/|H0|` (10.5× in CH, 400× in KdV).
- **Grace period** before H switches on, then `ε_rel = max(1/(epoch − grace), floor)`.
- **`ALM(penalty=0.)`** in both — see §5.3.
- `MAX_H=True` collapses the Nt per-slice constraints to their max.

`build_constraints` / `build_objective` / `plot_solution` / the Boole quadrature are duplicated
verbatim in both scripts (each stays standalone-runnable).

## 4. Per-script status

**Camassa–Holm** — the sane instance. `u_0 = 0.2 + 0.1cos2x` on `[-π, π]`, `tMax=5`, small smooth
data, wave speed ~0.25, so the solution travels ~1.2 over the run: about 8 cells of `dx=0.146`.
GELU + Xavier-normal, NAdam 5e-3 + ExponentialLR, 8000 epochs, `H_grace=3000`. BC penalises both
`u` and `u_x` periodicity.

**KdV** — pure translation of the TF script when `CONSTRAINED=False`; constrained mode carries two
documented structural traps (trivial-solution parking at `u≈0`, which is both an exact PDE solution
and a critical point of `H`; and `|H0| ≈ 2.4e-3` badly scaling the drift). Per-group dual rates
`H=1e-3`, `IC/BC=1e-1`. `2σ(x)−1` activation, Xavier-uniform, 5000 epochs, `H_GRACE=1000`. BC
penalises `u` only.

**2D** — an unfinished attempt to graft the CH constraint block onto the ZK case. Never ran.

## 5. Findings

### 5.1 `Nx=44` makes the H quadrature wrong at the level of the tolerance being chased — **FIXED 2026-09-09**

`H` uses composite Boole on the largest prefix with `(n−1) % 4 == 0` and **trapezoid on the
remainder**. With `Nx=44`, `(44−1) % 4 = 3`, so 3 of 43 intervals fall to the trapezoid tail. The
tail is a fixed number of intervals, so its absolute error is `O(h³)` and it dominates: the mixed
rule converges at order 3, not Boole's 6 *(verified: observed orders 3.89 → 3.05 over Nx = 44…704)*.

Relative error in `H0` against the exact integral *(verified, float64; float32 agrees to 6 digits)*:

| `Nx` | `(Nx−1) % 4` | CH | KdV (type 1) |
| --- | --- | --- | --- |
| 44 (**what the rule produces**) | 3 | 2.84e-04 | **4.18e-02** |
| 45 | 0 | 1.47e-16 | 7.99e-15 |

CH `H0 = 0.094221` vs exact `0.094248`. KdV `H0 = 2.4882e-3` vs exact `-νπ²/2 = 2.3884e-3`.

KdV's own tolerance floor is `H_EPS_FLOOR = 1e-2`, so the constraint is ~4× tighter than the
quantity it constrains can be measured. CH tightens to `ε_rel ≈ 2e-4` by epoch 8000, also at the
level of its quadrature bias. (Bias partly cancels between `H(u_θ)` and `H0` since both use the same
rule, but only partly — it depends on the shape of `u`.)

**Fixed** in both `Nx_from_arch` copies: `Nx += (-(Nx - 1)) % 4`, rounding up to the next `Nx` with
`(Nx−1) % 4 == 0`. `Nx = Nt` goes 44 → 45 and the collocation count 1936 → 2025 (against a target of
1976 and `Ntheta = 19761`, so the under-parameterized intent is preserved). Measured with the
scripts' own quadrature functions after the change *(verified)*:

| | before | after | floor |
| --- | --- | --- | --- |
| CH `H0` rel. err | 2.84e-04 | **1.70e-07** | float32 round-off |
| KdV type 1 `H0` rel. err | 4.18e-02 | **1.95e-06** | float32 round-off |

Both now sit at the float32 noise floor rather than at a discretization bias (float64 gives 1.5e-16
and 8.0e-15). The trapezoid-tail branch in `H_boole` / `H` is now unreachable at the default
architecture; it is kept for robustness under other `width`/`depth`/`fac`.

Smoke-tested at `Nx=45`: CH with `MAX_H=True` and `MAX_H=False`, KdV with `MAX_H=True`,
`MAX_H=False`, and `CONSTRAINED=False` — all five run to completion, and the 45-wide H constraint
block matches the dual count in the `MAX_H=False` paths.

### 5.2 The grid cannot resolve either KdV instance

- `type_pde=1` is Zabusky–Kruskal (`δ = 0.022`). The solution steepens by `t ≈ 1/π` and breaks into
  ~8 solitons with oscillation scale ~`δ = 0.022`, against `dx = 0.0465` and `dt = 0.116`, over
  `tMax = 5`. The structure is not representable on this grid at all.
- `type_pde=2`: `u_0 = 6 sech²x` on `[-20, 20]` → `dx = 0.91` against a soliton width of 1, i.e. one
  point per soliton; `tMax = 100`, `dt = 2.27`, while the wave travels at speed ≥ 12. Sharpened by
  the §5.1 fix: with the quadrature now exact, `H0` for this case is still **32% off** the true
  integral *(verified: -143.85 vs -211.20)*. That error is pure under-resolution, and it puts a
  hard floor under any drift tolerance this configuration could meaningfully enforce.

No amount of constraint or dual tuning fixes this. CH is the only well-resolved case, which is
consistent with it being the script that behaves.

Root cause: `fac=10` is an *optimization-theoretic* choice (stay in the interpolating regime, fewer
collocation points than parameters) that directly fights the *numerical* requirement of resolving
the solution. The two should be decoupled — see idea C.

### 5.3 `penalty=0.` means neither script is running an ALM

With `augmentation="quadratic"` and `penalty=0`, `_add_global_terms` returns early and
`_add_constraint_contributions` adds only `μᵀc`. The surrogate is `loss + μᵀc` and the dual step is
`μ ← clamp(μ + lr·c, 0, 1e4)` — plain gradient descent-ascent on the Lagrangian. For a nonconvex
problem that has no convergence guarantee, and the augmentation is exactly what is supposed to
supply one. Worth trying `penalty > 0`, and `augmentation="hpr"` (recently added to `alm.py`).

### 5.4 The 2D script is dead code

Crashes:
1. `custom_loss(xyt_train, model)` called with 2 args; signature is `(inputs, model, dual_opt)`.
   `dual_opt` is never defined and `dual_optim` is never imported.
2. `u_0_x` is not defined in this file.
3. `H(u_0(...), u_0_x(...), dx)` — the 2D `H` signature is `(u, u_x, u_y)`; `dx` lands in `u_y`.
4. `H(u_model.reshape(Nt, Nx), ...)` on a field of `Nx·Ny·Nt = 1000` elements.
5. `epoch` read as a global inside `custom_loss`.

Lines ~155–163 are verbatim copy-paste from the 1D CH script. Silent errors on top:

- `u_xyy = grad(u_y.sum(), y)` computes `u_yy`. The residual is therefore
  `u_t + u u_x + u_xxx + u_yy` — the wrong term, *not* a double count (`u_yy` appears only through
  this line). The TF original does it correctly: `u_xy = tape2.gradient(u_x, y)` then
  `u_xyy = tape.gradient(u_xy, y)`.
- Torch `H` sums over all three axes → a scalar. The TF `H` reduces axes `[0,1]` → one value per
  time slice. So `H_loss` min/max/mean are the same number and `std` of a 1-element tensor is `nan`;
  every H diagnostic in this script is vacuous.
- Torch `periodic_boundary_conditions` dropped the derivative terms — TF enforces
  `(uLx−uRx)² + (uLy−uRy)² + (uxL−uxR)² + (uyL−uyR)²`.
- `cheb_par` has `requires_grad=True` but is never handed to an optimizer, so the "trainable" ACS
  is not trainable. The plot at the end plots a scalar, not a history.
- `Nx=Ny=Nt=10` over `[0,8]²×[0,50]` → `dx=0.89`, `dt=5.6`, against solitons of width
  `sqrt(ε/c) ≈ 0.15`. Unresolvable by three orders of magnitude.
- `meshgrid(y, x, t, indexing='ij')` then `reshape(Nx, Ny, Nt)` has the first two axes swapped
  (harmless under a full sum, wrong for anything per-slice).
- The comment `input_dim = 3 + 4*2*4` says 35; the hardcoded 17 is the correct value for
  `FourierFeatures(n_modes=4)` (`[t]` + 4 modes × 4 = 17).

### 5.5 CH's IC/BC dual step size is inherited, not chosen

`add_constraint_group(m=1, is_ineq=True)` passes `lr=None`, `_drop_none` strips it, and
`Optimizer.add_param_group` then fills it from `self.defaults`, which `_scalar_defaults` built from
the *first* group. So all three CH groups run at `lr=5e-1`, the value picked for H. KdV sets both
rates explicitly. Since the abs-form fix removed the `1/|H0|` amplification a uniform rate may well
be right — but it should be a decision, not an inheritance.

### 5.6 Two constraints are stochastic and the rest are not

CH's `periodic_bc_loss` resamples 2000 random `t` every epoch, so its dual sees a noisy constraint
while H and IC are full-batch deterministic on a fixed grid. KdV's `periodic_bc` reuses the
collocation `t` and is deterministic. Pick one convention: either everything deterministic (and this
is a deterministic NLP, so §5.3 and idea E apply), or everything resampled (idea C).

### 5.7 Where the nonsmoothness actually is

- The `abs` kink at `H = H0` sits **strictly inside** the feasible set whenever `ε > 0`, where the
  dual is zero. Harmless at the solution; it only matters on transients that cross `H0`.
- `MAX_H=True` puts a `max` over time slices **on the active boundary** — the binding constraint is
  nondifferentiable exactly where it binds, and ALM will chatter between slices.

Options (see `FORMULATION.md` §7.4): keep the `Nt` smooth constraints (`MAX_H=False`, now
length-invariant); use
`(H−H0)² ≤ (ε|H0|)²`, smooth with the same feasible set; or split into the two one-sided smooth
constraints `H − H0 ≤ ε|H0|` and `H0 − H ≤ ε|H0|`.

### 5.8 There is no ground truth in any run — **FIXED 2026-09-09**

Both scripts now build a spectral reference at startup and report a true relative `L²` error
every `REF_EVERY` epochs (`FORMULATION.md` §11). The observation below stands as written; the
`type_pde=2` soliton is no longer the only route to a ground truth, and is now optional rather
than necessary.

Every diagnostic is self-referential: PDE residual, IC/BC mismatch, H drift. "Constraining `H`
gives a better solution" is not testable as set up. With the script's `type_pde=2` values
(`α=-3, ν=-1, ρ=0`) the equation *is* `u_t + 6u u_x + u_xxx = 0`, whose exact travelling wave is
`u = 2κ² sech²(κ(x − 4κ²t))`. The current `u_0 = 6 sech²(x)` is not one of them — matching requires
amplitude `2κ²` *and* width `1/κ` together. Setting `u_0 = 2κ² sech²(κx)` (with a domain and `tMax`
that resolve it) buys a real L² error curve for one line of change.

### 5.10 The invariant constraint was measuring a *t*-dependent error — **PARTLY FIXED 2026-09-09**

The root cause of the constrained runs converging to the frozen field, and the reason `penalty`,
dual rates and constraint forms were all red herrings.

The invariants were integrated on the **collocation** grid (`Nx = 45`). But the collocation grid is
sized from the parameter count (`fac = 10`) — an optimisation-theoretic choice — while a quadrature
grid has to resolve the invariant *density*, which is quadratic and cubic in `(u, u_x)` and so
spreads to far higher harmonics than `u` itself. Measured on the exact reference solution:

| | CH `h1` density | KdV `mom` density |
| --- | --- | --- |
| highest significant harmonic at `t=0` | 4 | 2 |
| at `t = t_max` | **128** | **~99** |
| collocation Nyquist (`Nx=45`) | 22 | 22 |

So by `t = t_max` roughly 2.4% of the CH `h1` density's energy is aliased. Pushing the **exact**
reference solution (true drift `~1e-10`) through the scripts' own quadrature returns:

| `NX_QUAD` | 45 | 89 | 181 | 361 | 721 |
| --- | --- | --- | --- | --- | --- |
| CH `h1` | 5.19e−02 | 1.05e−02 | 7.53e−04 | **4.00e−05** | — |
| CH `h2` | 5.99e−02 | 1.23e−02 | 8.14e−04 | 4.32e−05 | — |
| KdV `mom` | 2.12e−01 | — | 8.04e−04 | 7.40e−07 | 6.7e−12 |
| KdV `ham` | **1.09e+00** | — | 1.09e−02 | 4.07e−05 | 1.6e−11 |

At `Nx = 45` the KdV quadrature reports **109% drift in the Hamiltonian for the exact solution**.
Meanwhile the tolerance schedule tightens to `1e-4`, i.e. two to four orders of magnitude below the
measurement's own noise.

**Why that specifically produces the frozen field.** The aliasing error is *time-dependent* —
identically zero at `t=0`, since `u_0` is band-limited to harmonic 4 — and the constraint bounds
`I(t) − I(0)`, a *difference*. For a `t`-independent field the error is identical at every `t` and
cancels exactly. So `u(x,t) = u_0(x)` is the essentially unique feasible point at those tolerances,
and training converged to it correctly. It is not a local minimum; it is the answer to the question
that was asked.

Three things this rules out, each checked directly:

- *not the optimiser* — fitting the network to the reference by plain least squares reaches rel `L²`
  `0.0032` and still "violates" `h1` by 60×, and its drift does **not** improve as the fit improves
  (2.42e−2 at rel `L²` 0.0061, 2.45e−2 at 0.0032);
- *not the network* — the exact solution through the same quadrature scores slightly **worse**;
- *not `penalty=0`* — the frozen field is feasible, so the augmentation term is identically zero
  there; turning it on only makes the ALM converge to the frozen field more reliably.

It is also not a choice of rule: at `Nx=45` Boole gives `h1 = 5.19e-2`, the periodic trapezoid
`4.11e-2`, and `quad_rect` as written `4.28e-2`. No rule on 45 points can resolve harmonic 128.

**Fixed** by decoupling the two grids — `NX_QUAD = 361` in both scripts, with `I⁰` and `S` moved to
the same grid so the `t=0` cancellation is preserved — plus a startup check (`quadrature_floor`)
that pushes the reference through the script's own quadrature and refuses to start if any
`_EPS_FLOOR` sits below the resulting floor. Overhead is **14%** (16245 quadrature points against
2025 collocation points: the invariants need only a first derivative, the residual needs third and
mixed).

**Necessary but not sufficient — there are two floors, and only the first is fixed.** The mechanism
in the box above depends only on the *error being `t`-dependent*, not on its source. Quadrature
aliasing was one source; the network's own `∂_x u` error is another, and it survives the fix.
Measured by fitting `u_θ` to the reference on the decoupled grid *(verified)*:

| | drift floor | status |
| --- | --- | --- |
| quadrature aliasing (`N_x = 45`) | 5.19e−02 | **fixed** — now 4.0e−05 at `NX_QUAD = 361` |
| network `∂_x` error, at rel `L²` 0.0023 | **1.88e−02** | now binding |

The invariants depend on `u_x`, and a network's derivative error far exceeds its value error, so a
0.2% `L²` fit still misconserves `h1` by 2%. The decoupling did change the character of it — the
drift now *improves* with the fit (3.2e−2 at rel `L²` 0.0045 → 1.9e−2 at 0.0023) where on the
coupled grid it was flat (2.42e−2 → 2.45e−2) — but `1.9e-2` is still 20× above the `1e-3` the
schedule demands at 1500 epochs, so the frozen field remains uniquely feasible and a 1500-epoch run
still lands on it (rel `L²` 0.4215 vs the frozen 0.4222).

**Raising the floor above the measured value does break the degeneracy** *(verified)*: with
`_EPS_FLOOR = 5e-2` all three constraints are satisfied with duals ≈ 0 and the run reaches rel `L²`
**0.3999 — better than frozen** for the first time. It is still far from the 0.0023 the architecture
can represent, but that residual gap is ordinary PINN trainability, not a degeneracy.

Note the startup guard only checks the *quadrature* floor, so it passes at `_EPS_FLOOR = 1e-4` and
does not catch this second one. Extending it means fitting the reference at startup (~60 s).

This also does *not* fix KdV's representability problem (§5.2) — the network still cannot resolve 8
solitons with `K=8` and `Nx=45`. It means the constraint has stopped lying about it.

### 5.9 Minor

- `hist['H_self_err_max']` is allocated in CH and never appended to (its writer is commented out).
- `H_QUAD_RULE = 'rect'` is simply wrong on this grid: `quad_rect` sums all `Nx` points times `dx`,
  but `x_0` and `x_{Nx-1}` are the same physical point, so it double-counts one node — 3.1-4.1%
  error in every `I⁰` against 2e-16 for Boole. Not the default, so it has never been active.
- The unconstrained CH arm overfits the collocation x-grid by 14x: `mean(r²)` is 9.06e-04 on the
  training grid but 1.26e-02 on a 4x finer one (the constrained arm shows 1.0x — it is `u_0`, so
  there is nothing to alias). Residual comparisons between the arms are not like-for-like, and
  resampling collocation points each epoch would fix it.
- `loss_type='cs'` and `learning_rate_type='plateau'` in CH are stale filename labels: the
  aggregation is only a max when unconstrained, and the scheduler is `ExponentialLR`.
- `plt.show()` inside the CH training loop every 500 epochs blocks under an interactive backend.
- KdV's second `plot_solution` figure never gets `close()`d (harmless while `PLOT_EVERY=None`).
- `grad_L2_fft_batch` still scales by `L/Nx` instead of `L/Nx²`, so it overestimates `‖r_x‖²` by
  exactly `Nx` — inherited from TF, only rescales the tiny stabilisation weight `lam`.
- Torch CH has SoftAdapt commented out while the TF original has `do_training=True`; torch CH runs
  8000 epochs against the original's 1000.

## 6. Ideas worth trying

**A. Hard-encode the IC and BC instead of constraining them.** — **IMPLEMENTED 2026-09-09**
(`HARD_IC` / `HARD_BC` / `N_MODES` / `IC_TAU`; see `FORMULATION.md` §6.) Take
`u_θ(x,t) = u_0(x) + (1 − e^{−t})·N_θ(x,t)` for an exact IC, and Fourier features in `x` for exact
periodicity of `u`, `u_x` *and* `u_xx` (the 2D script already embeds `x, y` this way, and the TF 2D
has a commented `tf.where(t==0, u_0, u)` attempt at the IC half). The problem then collapses to
`min ‖r‖²  s.t.  |H − H0| ≤ ε|H0|`: one constraint group, no IC/BC dual tuning, and **no grace
period needed** — `u ≡ const` is unreachable once `u(·,0) = u_0` is structural, which is the trap
the grace period exists to dodge. A 3rd-order equation also *needs* three periodic conditions;
right now CH imposes two and KdV one.

**B. Constrain the hierarchy of invariants, low order first.** — **IMPLEMENTED 2026-09-09**
(`ACTIVE_INVARIANTS` / `EPS_SCALE`; see `FORMULATION.md` §7.) Both PDEs conserve more than `H`:

- CH: `∫u` (linear), `½∫(u² + u_x²)` (quadratic), `½∫(u³ + u u_x²)` (cubic — the one in use)
- KdV: `∫u` (Casimir of `∂_x`), `∫u²/2`, and `H`

The low-order ones are much better conditioned and nearly free to evaluate. For CH, `∫u` alone
breaks the trivial-solution trap: `∫u_0 = 0.4π ≈ 1.257` and `d(∫u)/dθ ≠ 0` at `u ≡ 0`, whereas the
cubic `H` has a gradient that vanishes to second order there. (For KdV type 1 the mass is exactly 0
and `d(∫u²)/dθ` also vanishes at `u ≡ 0`, so the grace period stays justified in that case.) As a
bonus this turns a one-constraint problem into a three-constraint one — a much better exercise of
the multi-group `dual_optim` machinery.

**C. Resample collocation points each step.** Solves three things at once: resolution without a
fixed 44² grid (§5.2), genuinely *stochastic* constraints so the stochastic dual theory in
`dual_optim` actually applies (§5.6), and it stops the network from gaming one fixed quadrature.
Practical shape: draw the `t` slices at random each step but keep a fine fixed `x` grid within each
slice, since `H` needs a spatial quadrature. The H constraint then becomes a random subsample of an
infinite family, which is exactly what nuPI/SSG are for.

**D. Tighten `ε` on progress, not on the epoch counter.** `ε = max(1/(epoch − grace), floor)`
tightens whether or not the current level was ever met. Classic continuation tightens only on
success and holds otherwise. Marge's penalty-barrier outer update already implements that rule.

**E. Run the repo's other solvers on this.** As written the problem is deterministic, small (≤1936
constraints, 19761 variables), nonconvex, and has max-type nonsmooth constraints — the target class
for `sqp/nonopt` and SPBM, with Marge covering the inequality/barrier side. Only ALM/iALM/nuPI are
wired in. A cross-solver comparison on one well-resolved instance (CH) would be a far stronger
result than more ALM tuning.

**F. Use named constraint groups.** `_gather_constraints` accepts a mapping keyed by group name, and
groups take `name=` / `bound=`. That retires `_ic_dual_idx` / `_bc_dual_idx` and the whole class of
silent group-misalignment bug CH already hit once. (`bound=` is static, so H's epoch-dependent `ε`
still needs either `group["bound"] = …` or the current value−tolerance encoding.)

**G. Factor the shared code out.** `build_constraints`, `build_objective`, `plot_solution` and the
Boole quadrature are duplicated verbatim across CH and KdV. The 2D script's breakage *is* a
copy-paste-drift failure — the same mechanism, one step further along.

## 7. Suggested order of work

1. ~~`Nx ≡ 1 (mod 4)` in `Nx_from_arch` (§5.1)~~ — **done 2026-09-09**, both scripts.
2. ~~Idea A (hard IC + Fourier BC)~~ — **done 2026-09-09**, both scripts. Removed two constraint
   groups, both IC/BC dual rates and the grace period from the tuning surface.
3. ~~Idea B (invariant hierarchy)~~ — **done 2026-09-09**, both scripts. Validated by a 1500-epoch
   CH run and a 1200-epoch KdV run (`FORMULATION.md` §9.1-9.2): the hard conditions hold at every
   epoch, the trap is gone in both, and CH converges to the continuation tolerance. **KdV plateaus
   infeasible** at `mean(r²) ≈ 0.16` — which is §5.2, not a defect in the formulation.
4. ~~A reference solution (§5.8)~~ — **done 2026-09-09**, both scripts (`FORMULATION.md` §11).
   Each solver verifies itself against the same invariant hierarchy the constraints use and aborts
   rather than serve as a bad baseline. First measurements on the saved 1000-epoch CH runs:
   unconstrained rel. `L²` = **0.620**, constrained = **0.422**, and the frozen field `u = u_0`
   scores **0.4222** — so the constrained run's entire gain is that it found the degenerate point
   of `FORMULATION.md` §6.8. It deviates from `u_0` by 0.0015 (0.7% of the IC amplitude) where the
   true solution deviates by 0.201. Also: the residual is *anti*-correlated with true error here
   (unconstrained `mean(r²)` = 9.1e-4 against constrained 5.5e-3), so ranking runs by `pde_loss`
   picks the worse one.
5. ~~Decouple the invariant quadrature grid from the collocation grid (§5.10)~~ — **done
   2026-09-09**, both scripts, plus a startup check. Removed the 5.19e−02 quadrature floor; exposed
   a 1.88e−02 network-derivative floor with the same consequence.
6. **Next:** set `_EPS_FLOOR` from the *total* measured floor rather than from `1e-4` (§5.10 —
   `5e-2` is verified to escape the frozen field), then attack the derivative floor itself, which is
   what §6.10's spectral output layer is for: exact `∂_x` by coefficient multiplication, and the
   quadratic invariants become exact algebra with no quadrature *and* no autograd derivative. Also
   re-run the §9 ablations — every drift number there predates this fix.
7. Only then revisit KdV, with a resolved grid (§5.2 — needs `K ≳ 91` and `Nx ≳ 182` for
   `type_pde=1`; §5.10 measured the same ~99 harmonics from the density side) and a ground-truth
   soliton (§5.8).
8. Rewrite or delete the 2D script; it cannot be incrementally fixed.
