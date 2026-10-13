# Structure-preserving PINNs — mathematical formulation

Companion to `CamassaHolm_trainableH1.py` and `kdv_trainableH1.py`, describing the formulation
implemented on 2026-09-09 (ideas A and B of `NOTES.md` §6). Numbers marked *(verified)* were computed
directly, not asserted; the verification scripts are described in §8.

---

## 1. Setting

Let `Ω = (x_min, x_max)` with period `L = x_max − x_min`, and `t ∈ [0, t_max]`. We seek
`u : Ω × [0, t_max] → ℝ` solving an evolution equation `𝓔[u] = 0` with

- initial condition `u(x, 0) = u_0(x)`,
- periodic boundary conditions in `x` (of as many orders as the equation requires),

represented by a neural surrogate `u_θ`, `θ ∈ ℝ^p`. Write `r(θ) := 𝓔[u_θ]` for the PDE residual.

Both target equations are Hamiltonian, so a functional `H[u]` is constant along exact solutions. The
premise of the experiment is that this is worth **imposing as a constraint** rather than adding to
the loss as another weighted term.

> **What the constraint can and cannot buy.** `H[u(·,t)] = H[u_0]` is a *consequence* of the PDE plus
> the IC, not independent information: at the exact solution the constraint is redundant. All of its
> value lies in the discrete, finite-capacity regime, where `r(θ) ≠ 0` and conservation is not
> implied. This is the PINN analogue of a structure-preserving integrator, and it is the claim the
> experiment has to support.

### 1.1 Notation

Three conventions carry most of the weight:

- **Square brackets mean *functional*.** `I[u]` eats an entire function and returns one scalar;
  `L(θ)` is an ordinary function of a finite vector. The distinction is the whole subject of §1.2:
  the conserved object is a functional of a whole frame, not a value at a point.
- **Subscripts on `u` are partial derivatives**, never indices: `u_x = ∂u/∂x`, `u_xxt = ∂³u/∂x²∂t`.
  Grid indices appear only on grid symbols (`x_i`, `t_j`) and invariant labels (`I_i`).
- **Calligraphic letters are operators**: `𝓔` the PDE, `𝒫` the Poisson operator, `𝓑` a boundary- or initial-condition
  operator, `𝓛` a lifting map.

**Domain, field, surrogate**

| symbol | meaning |
| --- | --- |
| `x`, `t` | space and time; `x ∈ Ω = (x_min, x_max)`, `t ∈ [0, t_max]` |
| `L` | spatial period, `L = x_max − x_min` |
| `u(x, t)` | the unknown scalar field — what the PDE is solved for |
| `u(·, t)` | one **frame**: the entire function of `x` at a fixed time `t` |
| `u_0(x)` | initial condition, `u(x, 0)`; `u_0′ = ∂_x u_0` |
| `u_t, u_x, u_xx, u_xxx, u_xxt` | partial derivatives of `u` |
| `m` | CH momentum variable, `m = u − u_xx` |
| `α, ρ, ν` | KdV coefficients (§2.2); `V(u) = αu³/3 + ρu²/2` its potential |
| `𝒜` | CH Helmholtz operator `1 − ∂_xx`, so `m = 𝒜u`; self-adjoint and positive |
| `θ ∈ ℝ^p` | network parameters; `p` = parameter count |
| `N_θ` | the raw MLP; `u_θ` the surrogate built from it by the ansatz (§6.2) |
| `𝓔[u]` | the PDE operator; the equation is `𝓔[u] = 0` |
| `r(θ)` | residual `𝓔[u_θ]` — itself a function of `(x, t)` |
| `‖r‖²_{H¹,h}` | the training objective (§4) |
| `v`, `d`, `ε` | generic perturbation function, parameter-space direction, small scalar |

**Functionals and Hamiltonian structure**

| symbol | meaning |
| --- | --- |
| `I[u]` | a functional: whole frame in, one scalar out |
| `I⁰` | its initial value `I[u_0]` — the number to be conserved |
| `⟨f, g⟩` | `L²` inner product `∫_Ω f g dx` |
| `δI/δu` | **functional derivative**: defined by `I[u+εv] = I[u] + ε⟨δI/δu, v⟩ + O(ε²)` |
| `H[u]` | the Hamiltonian — the functional that generates the dynamics |
| `𝒫` | Poisson operator, skew-adjoint; for KdV `𝒫 = ∂_x` |
| `∂_x`, `∂_x^n` | differentiation in `x` |
| `C[u]` | a **Casimir**: any functional with `𝒫(δC/δu) = 0` |
| `ker 𝒫` | kernel (null space) of `𝒫` |
| `{I = I⁰}` | level set to which the exact trajectory is confined |
| `q, p, q̇, ṗ` | canonical position/momentum, in the finite-dimensional analogy of §1.2 only — the one symbol clash in this document, since `p` is the parameter count everywhere else |

**The invariants used here** (`f_i` = density, so `I_i[u] = ∫_Ω f_i dx`)

| name | density `f_i` | equation |
| --- | --- | --- |
| `mass` | `u` | CH, KdV |
| `h1` | `½(u² + u_x²)` | CH |
| `h2` | `u³ + u u_x²` | CH |
| `mom` | `½u²` | KdV |
| `ham` | `V(u) − ½ν u_x²` | KdV |

**Discrete objects** (§4)

| symbol | meaning |
| --- | --- |
| `N_x`, `N_t` | collocation counts in `x` and `t`; `Δx = L/(N_x−1)`, `Δt = t_max/(N_t−1)` |
| `x_i`, `t_j` | grid nodes, `i = 0..N_x−1`, `j = 0..N_t−1` |
| `Q[f]` | composite Boole quadrature — the discrete stand-in for `∫_Ω · dx` |
| `I_i^h(t_j; θ)` | discrete invariant `i` at time `t_j`; `I_i⁰` its discrete reference |
| `k` | epoch index |
| `mean(r²)` | mean squared residual over the collocation grid |

**Constrained program and ansatz** (§5–§7)

| symbol | meaning |
| --- | --- |
| `𝒥` | active invariant set, e.g. `{mass, h1, h2}` — index set, distinct from the operator `𝒜` above |
| `S_i` | tolerance **scale** for invariant `i` (§7.3) |
| `ε_i(k)` | relative tolerance at epoch `k` (§7.2); constraint is `\|I_i^h − I_i⁰\| ≤ ε_i(k) S_i` |
| `μ_i` | dual variable (multiplier) of constraint group `i` |
| `𝓑[u] = g` | a linear condition to be imposed exactly; `u_p` a particular solution, `𝓛` the lift (§6.1) |
| `φ(t) = 1 − e^{−t/τ}` | IC ramp, `φ(0) = 0`; `τ` its timescale |
| `γ(x)` | periodic feature map `(sin 2πk x̃, cos 2πk x̃)_{k=1..K}`, `x̃ = (x − x_min)/L` |
| `K` | number of Fourier modes in `γ` |
| `HARD_IC`, `HARD_BC`, `MAX_H`, `EPS_SCALE` | script flags naming the choices above |

### 1.2 What an invariant is — for a reader coming from ML

Think of the solution as a movie: at each time `t` the frame `u(·,t)` is a whole function of `x`, and
the PDE says how one frame becomes the next.

An **invariant** (equivalently: conserved quantity, first integral) is a *functional* — something that
eats an entire frame and returns one scalar —

```
I : {functions of x} → ℝ,        I[u(·,t)] = I[u(·,0)]   for all t.
```

`u` itself changes a great deal; this one number does not move. Geometrically, the trajectory of the
dynamics is confined to the level set `{I = I⁰}`: it is a constraint the true dynamics satisfies
automatically, at no cost.

**Where they come from.** Write `⟨f,g⟩ = ∫_Ω f g dx`. The *functional derivative* `δH/δu` is the exact
analogue of a gradient — it is defined by the first-order expansion, just as `∇L` is:

```
H[u + εv] = H[u] + ε⟨δH/δu, v⟩ + O(ε²)     vs.     L(θ + εd) = L(θ) + ε⟨∇L, d⟩ + O(ε²)
```

KdV can be written `u_t = 𝒫(δH/δu)` with `𝒫 = ∂_x`, and `∂_x` is **skew**-adjoint on periodic
functions (`⟨f, ∂_x g⟩ = −⟨∂_x f, g⟩`, hence `⟨v, ∂_x v⟩ = ∫∂_x(v²/2) dx = 0`). Therefore

```
dH/dt = ⟨δH/δu, u_t⟩ = ⟨δH/δu, 𝒫 δH/δu⟩ = 0.
```

That is the entire mechanism, and the contrast with the optimisation we normally do is the point:

| | update | effect on the scalar |
| --- | --- | --- |
| gradient flow | `θ̇ = −∇L` | `L` decreases as fast as possible |
| Hamiltonian flow | `u̇ = 𝒫∇H`, `𝒫` skew | `H` is **exactly constant** |

Gradient descent moves *along* the gradient; Hamiltonian dynamics moves *perpendicular* to it,
circulating around level sets of `H` rather than descending them. In finite dimensions with
`𝒫 = [[0, I], [−I, 0]]` this is the familiar `q̇ = ∂H/∂p`, `ṗ = −∂H/∂q`.

**Two kinds, for two different reasons.** `H` is conserved because the flow is orthogonal to its own
gradient. A **Casimir** is conserved because its gradient lies in the *kernel* of `𝒫`: any `C` with
`𝒫(δC/δu) = 0`. For `𝒫 = ∂_x`, `C = ∫u` has `δC/δu = 1` and `∂_x 1 = 0`, so mass is conserved **for
any Hamiltonian** paired with this operator — it is a property of the geometry, not of the energy.
(Camassa–Holm has an analogous, in fact bi-Hamiltonian, structure; §3.1 proves its two low-order
invariants directly rather than deriving its bracket, and §3.3 verifies all three numerically.)

**They are equation-specific, not generic.** `∫u²` is conserved by KdV but *not* by CH, which needs
the `+ u_x²` term. §3.3 shows the same quantity drifting by `1.4e−15` under KdV and `4.1e−02` under
CH. Invariants cannot be guessed; they follow from the particular equation.

**What they are in this codebase.**

| symbol | plain reading |
| --- | --- |
| `mass = ∫u dx` | the area under the curve — the total "amount of stuff" |
| `mom = ½∫u² dx` (KdV) | the squared **L² norm** of the solution, exactly constant in time |
| `h1 = ½∫(u² + u_x²) dx` (CH) | the squared **H¹ Sobolev norm**: size plus wiggliness |
| `h2` / `ham` | the Hamiltonian itself — the functional generating the dynamics |

**Why they are worth constraining.** An invariant is a **self-supervision signal that needs no
labels**: we do not know the true `u`, but we know a property it must have, so `I = I⁰` is an
enforceable target without ground truth — which matters here precisely because there is none (§5.8).
Conservation is a *consequence* of the PDE, so an exact solver would get it for free; but the network
minimises a *pointwise* residual, and a small pointwise residual does not imply the *global* integral
is preserved, because errors can accumulate coherently. The invariant is a genuinely different,
global measure of wrongness that the residual does not see.

**Two things invariants are not.**

1. *Not sufficient.* The frozen field `u(x,t) = u_0(x)` conserves every invariant perfectly and is
   completely wrong — this is exactly the degenerate point analysed in §6.8. Conserving invariants
   removes a family of wrong answers; it does not identify the right one. A good drift number is not
   by itself evidence of a good solution.
2. *Not exactly what we constrain.* We constrain a **quadrature approximation** `I^h`, not the true
   `I` (§4). Conservation is therefore only meaningful to the accuracy of that quadrature — which is
   why the `N_x ≡ 1 (mod 4)` fix mattered (a 4.2% error against a 1% tolerance, `NOTES.md` §5.1), and
   why the spectral output layer of §6.10 is attractive: it makes the quadratic invariants exact
   algebra, with no quadrature at all.

---

## 2. The equations and their Hamiltonian structure

### 2.1 Camassa–Holm

```
u_t − u_xxt + 3u u_x − 2u_x u_xx − u u_xxx = 0
```

on `Ω = (−π, π)`, `t_max = 5`, with `u_0(x) = 0.2 + 0.1 cos 2x`.

Introducing the momentum `m := u − u_xx = 𝒜u` with `𝒜 := 1 − ∂_xx`, this is exactly the transport law

```
m_t + u m_x + 2 u_x m = 0,
```

since `m_t + u(u_x − u_xxx) + 2u_x(u − u_xx) = u_t − u_xxt + 3u u_x − 2u_x u_xx − u u_xxx`.

### 2.2 Korteweg–de Vries

```
u_t − α(u²)_x − ρ u_x − ν u_xxx = 0
```

with `V(u) := αu³/3 + ρu²/2` and

```
H[u] := ∫_Ω ( V(u) − ν u_x²/2 ) dx,        δH/δu = V′(u) + ν u_xx = αu² + ρu + ν u_xx =: w.
```

The equation is then precisely `u_t = ∂_x w = ∂_x (δH/δu)` — Hamiltonian with the Gardner Poisson
operator `∂_x`. Two configurations are provided:

| | `α` | `ρ` | `ν` | `Ω` | `t_max` | `u_0` |
| --- | --- | --- | --- | --- | --- | --- |
| `type_pde = 1` | −1/2 | 0 | −0.022² | (0, 2) | 5 | `cos πx` |
| `type_pde = 2` | −3 | 0 | −1 | (−20, 20) | 100 | `6 sech² x` |

`type_pde = 1` is the Zabusky–Kruskal problem `u_t + u u_x + δ² u_xxx = 0`, `δ = 0.022`.

---

## 3. The invariant hierarchies

### 3.1 Camassa–Holm

`𝒜` is self-adjoint and positive on periodic functions. Then

```
I₁[u] = ∫ m dx = ∫ u dx                  (linear in u)
I₂[u] = ½∫ u m dx = ½∫ (u² + u_x²) dx    (quadratic)
I₃[u] = ∫ (u³ + u u_x²) dx               (cubic)
```

are conserved. `I₃` is the classical Camassa–Holm Hamiltonian (Camassa & Holm 1993); `I₁` and `I₂`
follow in two lines each.

*`I₁`.* Since `u m_x + 2u_x m = ∂_x(u m) + u_x m`,

```
d/dt ∫m dx = −∫[∂_x(um) + u_x m] dx = −∫ u_x m dx
           = −∫ u_x(u − u_xx) dx = −∫ ∂_x(u²/2) dx + ∫ ∂_x(u_x²/2) dx = 0,
```

and `∫m dx = ∫u dx − ∫u_xx dx = ∫u dx` by periodicity.

*`I₂`.* Because `𝒜` is self-adjoint and commutes with `∂_t`, `∫ u_t m = ∫ u_t 𝒜u = ∫ u 𝒜u_t = ∫ u m_t`,
so `d/dt ∫ u m dx = 2∫ u m_t dx`. Then

```
∫ u m_t = −∫ u ∂_x(um) − ∫ u u_x m = +∫ u_x (um) − ∫ u u_x m = 0.
```

### 3.2 KdV

```
C₀[u] = ∫ u dx        (Casimir of ∂_x — linear)
C₁[u] = ½∫ u² dx      (momentum — quadratic)
H[u]  = ∫ (V(u) − ν u_x²/2) dx
```

*`C₀`.* `d/dt ∫u = ∫ ∂_x w = 0`.
*`C₁`.* `d/dt ½∫u² = ∫ u ∂_x w = −∫ u_x w = −∫ ∂_x(αu³/3 + ρu²/2 + ν u_x²/2) dx = 0`.
*`H`.* `d/dt H = ∫ (δH/δu) u_t = ∫ w ∂_x w = ∫ ∂_x(w²/2) dx = 0`.

### 3.3 Numerical confirmation *(verified)*

Spectral RK4, `N = 256` (CH) / `512` (KdV), integrating the equations exactly as the scripts write
them — same signs, same normalisations. Relative drift of each claimed invariant, with a
deliberately non-conserved control:

| | invariant | `I⁰` | rel. drift |
| --- | --- | --- | --- |
| CH, `t: 0 → 5` | `I₁ = ∫u` | +1.2566370614e+00 | 1.1e−15 |
| | `I₂ = ½∫(u²+u_x²)` | +2.0420352248e−01 | 2.6e−10 |
| | `I₃ = ∫(u³+u u_x²)` | +9.4247779608e−02 | 4.7e−10 |
| | *control* `∫u²` | +2.8274333882e−01 | **4.1e−02** |
| KdV t1, `t: 0 → 0.2` | `C₀ = ∫u` | −1.7e−16 | (degenerate, see §7.3) |
| | `C₁ = ½∫u²` | +5.0000000000e−01 | 1.4e−15 |
| | `H` | +2.3884442651e−03 | 9.5e−14 |
| | *control* `∫u³` | −5.6e−17 | **O(1)** |

The controls drift by 4% and O(1), so the test discriminates rather than passing trivially.
`I₃⁰ = 0.09424778` also matches the reference value the script computes on its own grid, and
`H⁰ = 2.3884e−3 = −νπ²/2` exactly, confirming both sign conventions.

---

## 4. Discretisation

**Collocation.** Tensor grid `x_i = x_min + iΔx` (`i = 0..N_x−1`, `Δx = L/(N_x−1)`) and
`t_j = jΔt` (`j = 0..N_t−1`, `Δt = t_max/(N_t−1)`), with `N_x = N_t` fixed by the parameter count,
`N_x = ⌈√(p/fac)⌉` rounded up to satisfy `(N_x − 1) ≡ 0 (mod 4)`. At `width = 80`, `depth = 4`,
`fac = 10` this gives `N_x = N_t = 45`, i.e. 2025 collocation points *(verified)*. Both endpoints are
included, so `x_0` and `x_{N_x−1}` are the same physical point of the periodic domain — which is what
makes a closed-interval rule the right quadrature for one period.

**Quadrature.** Composite Boole,

```
Q[f] := (2Δx/45) Σ_b ( 7f_{4b} + 32f_{4b+1} + 12f_{4b+2} + 32f_{4b+3} + 7f_{4b+4} ),
```

exact for polynomials of degree ≤ 5, and requiring `(N_x − 1) ≡ 0 (mod 4)` — the reason for the
rounding above. Off that residue class the leftover nodes fall to a trapezoid tail whose *fixed*
interval count makes the composite rule `O(Δx³)`; see `NOTES.md` §5.1.

**Discrete invariants.** For each active invariant `i` and each time index `j`,

```
I_i^h(t_j; θ) := Q[ f_i( u_θ(·,t_j), ∂_x u_θ(·,t_j) ) ],       I_i⁰ := Q[ f_i(u_0, u_0′) ].
```

The reference uses the *same* rule as the constraint, so the systematic quadrature bias cancels
exactly at `t = 0`. **It does not cancel for `t > 0`**, and that was fatal: see §12, which is also
why `Q` is no longer evaluated on the collocation grid.

**Residual norm.** `‖r‖²_{H¹,h}` is the mean-square residual plus a ramped gradient term:

- CH: `mean(r²) + λ(k)·mean(r_x²)`, `λ(k) = 1 − exp(−10⁻³ max(k − 1000, 0))`;
- KdV: `mean(r²) + λ(k)·mean_j ‖∂_x r(·,t_j)‖²_spectral`, `λ(k) = 0.01 Δx² min(1, k/1000)`,

with `k` the epoch. (The KdV spectral norm still carries the inherited factor-of-`N_x` normalisation
error noted in `NOTES.md` §5.9; it only rescales `λ`.)

---

## 5. The constrained program

### 5.1 Before

```
min_θ   ‖r(θ)‖²_{H¹,h}
s.t.    |H^h(t_j;θ) − H⁰| ≤ ε(k)|H⁰|,   j = 1..N_t   (or one constraint on max_j)
        MSE_IC(θ) ≤ 10⁻⁶
        MSE_BC(θ) ≤ 10⁻⁶
```

Three dual groups, `[H, IC, BC]`, with the layout maintained by hand-written offsets.

### 5.2 After

The IC and the periodic BC now hold identically in `θ` (§6), so they leave the constraint set
entirely, and the invariant hierarchy replaces the single Hamiltonian:

```
min_θ   ‖r(θ)‖²_{H¹,h}
s.t.    | I_i^h(t_j;θ) − I_i⁰ |  ≤  ε_i(k) · S_i ,     i ∈ 𝒥,  j = 1..N_t
```

with `𝒥 ⊆ {mass, h1, h2}` (CH) or `{mass, mom, ham}` (KdV), one dual group per `i`, and `S_i` the
tolerance scale of §7.3. Under `MAX_H` the `N_t` constraints of each group collapse to their maximum.

---

## 6. Idea A — hard-encoded initial and boundary conditions

### 6.1 The general pattern: constraint lifting

To impose a *linear* condition `𝓑[u] = g` exactly rather than by penalty, write

```
u  =  u_p  +  𝓛[v],        𝓑[u_p] = g,      range(𝓛) ⊆ ker(𝓑),
```

so `𝓑[u] = g` holds identically in `v`. The condition leaves the optimisation problem entirely: no
penalty weight, no multiplier, no tolerance. Two things must be checked — that `range(𝓛) ⊆ ker(𝓑)`
(*exactness*, §6.3–6.4) and that `range(𝓛)` is dense in `ker(𝓑)` (*completeness*, §6.5). The second is
the one usually skipped, and it is where a badly chosen lift silently removes the true solution from
the hypothesis class.

Both of our conditions are linear in `u`, so both admit a lift:

| condition | `𝓑[u]` | `g` | `u_p` | `ker 𝓑` | `𝓛` |
| --- | --- | --- | --- | --- | --- |
| IC | `u(·,0)` | `u_0` | `u_0(x)` (constant in `t`) | `{v : v(·,0) = 0}` | `v ↦ φ(t)·v`, `φ(0)=0` |
| periodic BC | `(∂_x^n u(x_min,·) − ∂_x^n u(x_max,·))_n` | `0` | `0` | `⊇ {L-periodic}` | precomposition with `γ` |

### 6.2 The ansatz

```
u_θ(x, t)  =  u_0(x)  +  φ(t) · N_θ( γ(x), t ),        φ(t) = 1 − e^{−t/τ}
γ(x)       =  ( sin(2πk x̃), cos(2πk x̃) )_{k=1..K},     x̃ = (x − x_min)/L
```

`N_θ : ℝ^{2K+1} → ℝ` is the MLP, which never sees raw `x`. Defaults `K = 8`, `τ = 1`. The two lifts
compose because they act on different arguments: `φ(t)` handles the IC, `γ(x)` the BC, and neither
disturbs the other.

### 6.3 Exactness of the initial condition

`φ(0) = 1 − e⁰ = 0` — and *exactly* zero in floating point, since `exp(0)` returns `1.0` and
`1.0 − 1.0 = 0.0` with no rounding. Hence `u_θ(x,0) = u_0(x) + 0·N_θ = u_0(x)` for every `θ`, bit for
bit. *(verified: residual `0.000e+00` for CH and KdV `type_pde=1`, at initialisation and at every one
of 1500 / 1200 training epochs.)*

For `type_pde=2` the check reads `1.2e−07` *relative*, which is one float32 ulp. It is not a property
of the ansatz: `φ(0)` is still exactly `0`, and the culprit is that `u_0` is evaluated once on the
strided view `xt[:, 0:1]` inside `forward` and once on a contiguous tensor in the check, and
`torch.cosh` takes a different kernel path on a non-contiguous input. `0.0 · N` contributes nothing.

### 6.4 Exactness of periodicity, to all orders

Each feature satisfies `γ_j(x + L) = γ_j(x)`, since the argument advances by exactly `2πk`. Therefore
`x ↦ N_θ(γ(x), t)` is `L`-periodic for any `N_θ`, and — this is the part that matters — **so is every
`x`-derivative**: by the chain rule `∂_x^n (N_θ ∘ γ)` is a finite sum of products of partial
derivatives of `N_θ` evaluated at `γ(x)` with derivatives of the `γ_j`, each factor `L`-periodic.
`u_0` is itself `L`-periodic in both problems (`0.2 + 0.1cos2x` has period `π`, which divides `2π`;
`cos πx` has period `2 = L`). Hence

```
∂_x^n u_θ(x_min, t) = ∂_x^n u_θ(x_max, t)      for every n ≥ 0 and every t,
```

*(verified: `max_t |u_θ(x_min,t) − u_θ(x_max,t)| = 1.5e−07`, float32 round-off, sustained through
training with `BC MSE ~ 5e−15`.)*

This is a strict gain in what is imposed, not just in how. Both equations are third order and so
require three periodic conditions; the soft CH version penalised two (`u`, `u_x`) and the soft KdV
version one (`u`). The remaining orders were simply never imposed — the plotting code in
`kdv_trainableH1.py` still carries a comment saying the `u_x` panel is "a pure diagnostic — nothing in
the loss drives it to zero". Now all orders hold identically.

It also removes an inconsistency noted in `NOTES.md` §5.6: CH's `periodic_bc_loss` resampled 2000
random times every epoch, so that one constraint was *stochastic* while everything else was
full-batch deterministic. Exact periodicity for all `t` retires the question.

### 6.5 Completeness — does hard-encoding shrink the hypothesis class?

This is the claim worth checking, because a lift that is exact but not complete would quietly make the
true solution unreachable.

**The IC lift is complete iff `φ` has a simple zero at `0`.** Let `u*` be any target with
`u*(·,0) = u_0`. Solving the ansatz for `N` gives

```
N(x, t) = ( u*(x, t) − u_0(x) ) / φ(t).
```

The numerator vanishes at `t = 0`, so this is `0/0`; it extends continuously iff the numerator's zero
is at least the order of `φ`'s. For `u*` that is `C¹` in `t` the numerator vanishes to first order, and
`φ(t) = t/τ + O(t²)`, so

```
N(x, 0⁺) = τ · ∂_t u*(x, 0),
```

finite. Had `φ` a *double* zero (say `φ = t²`), completeness would fail unless `∂_t u*(·,0) ≡ 0` — the
ansatz would exclude every solution with a nonzero initial time derivative. So `φ(0) = 0` buys
exactness and `φ′(0) ≠ 0` buys completeness; both are required.

*Verified along the true CH solution* (spectral RK4, `τ = 1`), forming `N = (u* − u_0)/φ` explicitly:

| `t` | `φ(t)` | `max\|u* − u_0\|` | `max\|N\|` | `max\|N − τ ∂_t u*(·,0)\|` |
| --- | --- | --- | --- | --- |
| 1e−04 | 1.000e−04 | 5.849e−06 | 0.0585 | 3.707e−06 |
| 1e−03 | 9.995e−04 | 5.849e−05 | 0.0585 | 3.707e−05 |
| 1e−02 | 9.950e−03 | 5.853e−04 | 0.0588 | 3.715e−04 |
| 0.1 | 9.516e−02 | 5.888e−03 | 0.0619 | 3.789e−03 |
| 1.0 | 6.321e−01 | 6.127e−02 | 0.0969 | 4.503e−02 |
| 5.0 | 9.933e−01 | 1.951e−01 | 0.1965 | 2.091e−01 |

The `0/0` cancels cleanly: `max|N|` stays in `[0.059, 0.197]` across four decades of `t`, and the last
column falls linearly in `t`, confirming `N → τ ∂_t u*(·,0)` at the predicted first order. So the
function the network is actually asked to learn is bounded, `O(0.1)`, and smooth — the lift is not
merely complete but *well-conditioned* for this problem. Note also that an error `δ` in `N` produces
an error `φ(t)δ ≤ δ` in `u`, so the lift **damps** approximation error near `t = 0` rather than
amplifying it.

**The BC lift is complete already at `K = 1`.** `γ_1(x) = (sin 2πx̃, cos 2πx̃)` maps `[x_min, x_max)`
bijectively onto the unit circle `S¹ ⊂ ℝ²`. Any continuous `L`-periodic `f` therefore factors as
`f = f̃ ∘ γ_1` with `f̃` continuous on `S¹`; extend `f̃` to a continuous `F` on a neighbourhood (Tietze)
and approximate `F` by an MLP on that compact set. Hence `{N_θ ∘ γ_1}` is already dense in the
continuous `L`-periodic functions, and for a third-order PDE the same holds in `C³` for smooth
activations (Hornik 1991, on derivative approximation). **`K > 1` therefore buys no expressiveness in
the limit** — it changes conditioning and spectral bias, which is the subject of §6.7.

### 6.6 The choice of `φ` and `τ`

`φ(t) = t` satisfies both requirements of §6.5 but is unbounded on `[0, t_max]`: `N` must be
`O(1/t_max)` to produce `O(1)` deviations, and `u_t = N + t N_t` retains a persistent `N` term.
`φ(t) = 1 − e^{−t/τ}` is bounded in `[0,1)` with `φ′(t) = e^{−t/τ}/τ → 0`, so at large `t`,
`u ≈ u_0 + N` and `u_t ≈ N_t`: the ansatz becomes transparent and hands over to the network.

`τ` is not free — the PDE *pins* `N(·,0)`. Since `N(x,0) = τ ∂_t u(x,0)` and the equation determines
`∂_t u(·,0)` from `u_0`, *(verified)*

| | `max\|∂_t u(·,0)\|` | `max\|N(·,0)\|` at `τ = 1` |
| --- | --- | --- |
| CH | 0.0585 | 0.0585 |
| KdV `type_pde=1` | 1.5814 | 1.5814 |

both `O(1)` or below, so `τ = 1` needs no scale compensation. `τ ≪ 1` keeps `N(·,0)` small but makes
`φ′(0) = 1/τ` large, concentrating the transition in a thin layer near `t = 0` that the network must
resolve; `τ ≫ t_max` degenerates to `φ ≈ t/τ` and reintroduces the scale problem of `φ = t`.

### 6.7 The choice of `K` — resolution against conditioning

Two pressures act in opposite directions.

*Resolution* wants `K` large enough to reach the highest harmonic the solution carries. CH's `u_0`
excites mode 2 and the solution stays smooth and low-mode, so `K = 8` is ample. KdV `type_pde=1`
develops dispersive oscillations at harmonic index `≈ L/δ = 91`.

*Conditioning* wants `K` small. The `n`-th `x`-derivative of the `k`-th feature carries a factor
`(2πk/L)^n`, and a third-order residual involves the cube *(verified)*:

| | `K = 1` | `K = 8` | `K = 16` | `K = 91` |
| --- | --- | --- | --- | --- |
| CH (`L = 2π`), `∂_x³` factor | 1 | 512 | 4 096 | 7.5e+05 |
| KdV (`L = 2`), `∂_x³` factor | 31 | 1.59e+04 | 1.27e+05 | 2.34e+07 |

So the residual's sensitivity along the highest-mode path grows like `K³`. Since §6.5 showed that
`K > 1` adds nothing to expressiveness, `K` is *purely* a conditioning and spectral-bias knob — and
raising it to resolve KdV `type_pde=1` would inflate the third-derivative scale by seven orders of
magnitude. Note also that the domain length enters: `L = 2` gives `2πk/L = πk`, making the same `K`
about `π` times worse for KdV than for CH.

This is an independent restatement of `NOTES.md` §5.2, from the ansatz side rather than the grid side:
KdV `type_pde=1` is not merely under-resolved on a 45-point grid, it is at a `K` where resolving it and
conditioning it pull hard against each other.

### 6.8 Why this removes the trivial-solution trap

**The old trap.** `u ≡ 0` was simultaneously

1. *an exact solution of the PDE* — every term of `r` is quadratic or higher, or a pure derivative, so
   `r[0] = 0` and the **objective is globally minimised** there; and
2. *a critical point of the constrained invariant*, so the one thing pushing away from it had a
   vanishing gradient.

The degeneracy in (2) is exactly quantifiable. For `u = εv` and an invariant whose density is
homogeneous of degree `q` in `(u, u_x)`, `I[εv] = ε^q I[v]` and

```
∇_θ | I − I⁰ |  =  sign(·) ∫ (δI/δu) ∂_θ u  =  O(ε^{q−1}).
```

So the order of degeneracy at `u ≡ 0` is `q − 1`:

| invariant | degree `q` | `δI/δu` | `∇_θ` at `u = εv` |
| --- | --- | --- | --- |
| CH `I₁ = ∫u`, KdV `C₀` | 1 | `1` | **`O(1)` — never degenerates** |
| CH `I₂`, KdV `C₁` | 2 | `u − u_xx` / `u` | `O(ε)` |
| KdV `H` | 2 and 3 mixed | `αu² + ρu + ν u_xx` | `O(ε)` (via the dispersive term) |
| CH `I₃` | 3 | `3u² − u_x² − 2u u_xx` | `O(ε²)` — worst |

This is the precise sense in which the low-order members are better conditioned, and it is why the
Hamiltonian alone was the hardest possible choice of single constraint.

Note the asymmetry it implies for the two problems. For CH, `I₁⁰ = 0.4π ≈ 1.2566 ≠ 0`, so `u ≡ 0`
violates the mass constraint by `1.2566` with an `O(1)` gradient: **the mass constraint alone breaks
the trap**, no grace period required. For KdV `type_pde=1`, `C₀⁰ = ∫₀² cos πx dx = 0` exactly, so
`u ≡ 0` *satisfies* mass, and `C₁`'s gradient is `O(ε)` — mass does not break the KdV trap. Hence the
implementation sets `GRACE = 0 if HARD_IC else 1000`.

**Under the ansatz.** `u_θ(·,0) = u_0 ≢ const`, so `u ≡ const` is not in the model class at all. The
nearest degenerate point is `N_θ ≡ 0`, i.e. the frozen field `u_θ(x,t) = u_0(x)`. There:

- every invariant is *exactly* conserved (`I_i^h(t_j) = I_i⁰` for all `j`), so the point is strictly
  feasible and all duals relax to zero; but
- the PDE residual is large: `mean(r²) = 5.05e−02` for CH and `1.23e+00` for KdV `type_pde=1`
  *(verified)*.

So the degenerate point is **feasible but objective-suboptimal**, and the objective gradient points
away from it — as against the old trap, which was objective-*optimal* and only weakly infeasible.
That is the difference between an attractor and a non-attractor, and it is the structural reason the
grace period is no longer needed.

### 6.9 What it costs, and when it is inappropriate

The periodic embedding raises the input dimension from 2 to `2K + 1 = 17`, so the parameter count goes
from 19 761 to 20 961 and the grid-sizing rule returns the same `N_x = N_t = 45` *(verified)*. Runtime
cost is the embedding plus a wider first layer — negligible against the third-derivative autograd.

Two genuine limitations:

1. **The network can no longer represent a non-periodic component.** Periodicity is hard-coded as a
   modelling assumption, so one can no longer ask whether the network *discovers* it. For these
   problems periodicity is given, so nothing is lost; for a problem where the boundary behaviour is
   itself in question, this lift is the wrong tool.
2. **Hard encoding is only right for exact, noise-free conditions.** A soft IC lets the fit trade
   accuracy in `u_0` against accuracy in the residual, which is exactly what you want when `u_0` is
   observational data. Here `u_0` is an analytic expression, so the trade-off has no value and giving
   it up is free. With noisy initial data one should go back to the constrained (or penalised) form.

Both scripts keep `HARD_IC` / `HARD_BC` as flags precisely so the soft formulation remains available
as the ablation baseline.

### 6.10 Alternatives, and one that is strictly better

- **`where(t == 0, u_0, u)`** — commented out in `2D_trainableACS_tf.py`. This sets the value at `t = 0`
  but is discontinuous in `t` and has zero gradient almost everywhere, so it is useless for an
  equation containing `u_t`.
- **Distance-function lifting** (Sukumar & Srivastava): `u = g + D·N` with `D` vanishing on the
  boundary. `φ(t)` *is* this construction with `D = φ`, and §6.5 is the statement that `D` must have a
  simple, not higher-order, zero.
- **A spectral output layer** — strictly stronger, and the natural next step:

  ```
  u_θ(x,t) = a_0(t) + Σ_{k=1..K} [ a_k(t) cos(2πk x̃) + b_k(t) sin(2πk x̃) ]
  ```

  with `(a_k, b_k)` the outputs of a network of `t` alone. This keeps exact periodicity in all orders,
  and adds three things the present ansatz does not have:

  1. **`∂_x^n u` becomes exact.** Differentiation in `x` is multiplication of the coefficients by
     `(2πk/L)^n` — no autograd through a nonlinearity in `x` at all. The `K³` factor of §6.7 becomes an
     explicit, inspectable coefficient rather than a gradient path.
  2. **The quadratic invariants become exact algebraic functions of the coefficients**, by
     orthogonality:
     ```
     ∫ u dx   = a_0 L
     ∫ u² dx  = a_0² L + (L/2) Σ_{k≥1} (a_k² + b_k²)
     ∫ u_x² dx =        (L/2) Σ_{k≥1} (2πk/L)² (a_k² + b_k²)
     ```
     so CH's `mass` and `h1`, and KdV's `mass` and `mom`, need **no quadrature** — which removes the
     entire §5.1 quadrature-error concern for them and makes their reference values exact.
  3. Only the cubic invariant (`h2` / `ham`) still needs a quadrature, or a dealiased convolution.

  The cost is conceptual: it is a spectral-in-`x`, neural-in-`t` method rather than a mesh-free PINN,
  which changes what the experiment is claiming. Worth doing as a second arm, not a replacement.

## 7. Idea B — constraining the invariant hierarchy

### 7.1 Constraint form

```
c_{i,j}(θ) = | I_i^h(t_j;θ) − I_i⁰ | − ε_i(k) · S_i  ≤  0
```

**Absolute drift with a scaled tolerance, not relative drift.** The feasible sets of
`|ΔI| ≤ ε S` and `|ΔI|/S ≤ ε` are identical, but

```
∇_θ ( |ΔI| / S )  =  (1/S) ∇_θ |ΔI| ,
```

so the relative form amplifies that group's gradient by `1/S`. With `S = |H⁰| = 2.39e−3` for KdV
that is a **419×** amplification of the Hamiltonian term relative to the others, which is what let
the `H` multiplier outrun the IC and BC groups in the earlier runs. Putting the scale in the
tolerance removes the amplification while keeping the constraint dimensionless in `ε`. (Applying
*both* — dividing the drift *and* scaling the tolerance — makes the bound `1/S` times too tight; this
is flagged in both scripts because it was a real bug once.)

### 7.2 Tolerance schedule

```
ε_i(k) = max( 1/(k − k_grace), ε_i^floor )   for k > k_grace,    constraint inactive otherwise.
```

A continuation: loose at switch-on, tightening toward the floor. Each group carries its own floor and
its own dual step size. (`NOTES.md` §6D argues this should tighten on *progress* rather than on the
epoch counter; not yet implemented.)

### 7.3 The tolerance scale `S_i`

Two options, selected by `EPS_SCALE`:

- `'l1'` (default): `S_i = Q[ |f_i(u_0, u_0′)| ]`, the L¹ norm of the density;
- `'value'`: `S_i = |I_i⁰|`, the conventional "relative error in the invariant".

They **coincide exactly whenever the density has one sign**. Measured on the actual `N_x = 45` grid
*(verified)*:

| | `I⁰` | `S = L¹` | `\|I⁰\|/S` |
| --- | --- | --- | --- |
| CH `mass` | +1.2566370614e+00 | 1.2566370614e+00 | 1.0000 |
| CH `h1` | +2.0420352248e−01 | 2.0420352248e−01 | 1.0000 |
| CH `h2` | +9.4247779608e−02 | 9.4247779608e−02 | 1.0000 |
| KdV `mass` | −6.4e−17 | 1.2683299114e+00 | **5.0e−17** |
| KdV `mom` | +5.0000000000e−01 | 5.0000000000e−01 | 1.0000 |
| KdV `ham` | +2.3884442651e−03 | 1.4197554833e−01 | **1.7e−02** |

For CH every density is one-signed (`u_0 = 0.2 + 0.1cos2x > 0`), so the choice is immaterial. For KdV
it is not:

- **`mass` is degenerate under `'value'`.** `∫₀² cos πx dx = 0` exactly, so `S = 0` and the constraint
  would silently demand exact conservation. Worse, the degeneracy is invisible to a naive test: in
  float32 the integral evaluates to `4.8e−08`, not `0`, so a `S <= 0` guard passes it through and the
  constraint quietly asks for conservation to eight digits. The implementation therefore tests
  degeneracy **relative to the L¹ norm**, `S ≤ 10⁻⁶·L¹`, which separates `5.0e−17` from the
  Hamiltonian's legitimate `1.7e−02` by ten orders of magnitude, and raises rather than proceeding.
- **`ham` is badly scaled under `'value'`.** `|H⁰| = 2.39e−03` is small only because `∫cos³ = 0` kills
  the cubic part, leaving pure dispersion `−νπ²/2`, while the density being integrated is `O(0.14)`.
  Scaling by `|H⁰|` therefore asks for the *cancellation* to be preserved, not merely for the drift to
  be small relative to the terms contributing to it.

Both numbers are reported every epoch: `drift/S` drives the constraint, and `drift/|I⁰|` is logged
alongside it (`NaN` where `I⁰ = 0`) so the conventional figure stays available for comparison with
the literature.

### 7.4 Where the nonsmoothness is

`|·|` is not differentiable at `I = I⁰`, but with `ε_i > 0` that kink lies **strictly inside** the
feasible set, where the multiplier is zero — so it is harmless at the solution and only matters on
transients that cross `I⁰`. The `max_j` under `MAX_H`, by contrast, sits **on the active boundary**:
the binding constraint is nondifferentiable exactly where it binds. Smooth alternatives with the same
feasible set: keep the `N_t` constraints (`MAX_H = False`), use `(ΔI)² ≤ (ε S)²`, or split into the
two one-sided constraints `ΔI ≤ εS` and `−ΔI ≤ εS`.

---

## 8. The dual method, and one caveat

With `𝒢` the registered groups,

```
L(θ, μ) = ‖r(θ)‖²_{H¹,h} + Σ_{g∈𝒢} μ_g ᵗ c_g(θ),
μ_g  ←  Π_{[0, 10⁴]} ( μ_g + η_g c_g(θ) ),
```

each group having its own step size `η_g`. Constraints are handed to `dual_optim` as a **list of
per-group tensors** rather than one flat vector, so the optimiser validates each group's arity; a
layout slip now raises instead of feeding one group's values onto another's duals. Group registration
and constraint assembly both iterate the same `GROUPS` list, so the two cannot drift apart — this is
the failure mode the earlier hand-written offsets had, where disabling the IC group silently shifted
the BC values onto the wrong multipliers.

**Caveat.** Both scripts still construct `ALM(..., penalty=0.)`, which makes the surrogate a plain
Lagrangian and the update projected gradient descent-ascent — *not* an augmented Lagrangian. For a
nonconvex problem that has no convergence guarantee, and the augmentation is precisely what is meant
to supply one. See `NOTES.md` §5.3; unchanged by this work.

---

## 9. Summary of the change

| | before | after |
| --- | --- | --- |
| IC | constrained, `MSE ≤ 10⁻⁶` | **exact by construction** |
| periodic BC | penalised/constrained, `u` and `u_x` (CH) or `u` only (KdV) | **exact, all orders** |
| constraints | 1 invariant (`H`) | 1–3 invariants, low order first |
| dual groups | 3 (`H`, `IC`, `BC`) | `\|𝒥\|` (3 by default) |
| group layout | hand-written offsets | one `GROUPS` list drives both |
| grace period | required (trap at `u ≡ 0`) | `0` (trap not representable) |
| worst degeneracy | `O(ε²)` (cubic `H` alone) | `O(1)` (linear mass present) |
| tolerance scale | `\|H⁰\|`, degenerate when `H⁰ ≈ 0` | L¹ of the density, guarded |
| objective | PDE residual | PDE residual (unchanged) |

Coverage: 12 CH and 8 KdV flag combinations run to completion, including `MAX_H = False` (which widens
each invariant block to 45), every subset of the hierarchy, both `HARD_*` flags in all four
combinations, `CONSTRAINED = False`, `type_pde ∈ {1,2}`, both quadrature rules, and both `EPS_SCALE`
settings *(verified)*.

### 9.1 A 1500-epoch Camassa–Holm run *(verified)*

Hard IC + BC, all three invariants constrained, `GRACE = 0`, 178 s:

| epoch | objective | IC MSE | BC MSE | `mass` drift (μ) | `h1` drift (μ) | `h2` drift (μ) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.67e+01 | 0 | 5.9e−14 | 2.4e−01 (0) | 1.3e+00 (0) | 1.8e+00 (0) |
| 201 | 4.88e−02 | 0 | 4.8e−15 | 3.0e−03 (9.8) | 2.6e−03 (1.6) | 4.2e−03 (1.4) |
| 1001 | 9.59e−04 | 0 | 4.7e−15 | 1.2e−02 (13.6) | 1.7e−02 (2.4) | 2.6e−02 (2.0) |
| 1500 | 4.59e−02 | 0 | 4.7e−15 | 9.7e−04 (15.7) | 5.3e−04 (2.8) | 8.4e−04 (2.4) |

What this confirms:

1. **The hard conditions hold throughout training, not merely at initialisation.** `IC MSE` is
   identically `0` at every epoch and `BC MSE` stays at `~5e−15` (i.e. `|u(x_min,t) − u(x_max,t)| ≈
   7e−8`, float32 round-off). Neither is being trained — they are diagnostics that verify the ansatz.
2. **No trivial-solution trap.** The objective descends from `26.7` to `~1e−3` with no stalled phase.
   The old formulation parked at `u ≈ 0` with the data term frozen.
3. **The multiplier ordering matches the degeneracy analysis of §6.8.** `μ_mass = 15.7 ≫ μ_h1 = 2.83 >
   μ_h2 = 2.39`: the linear invariant, whose constraint gradient never degenerates, is the one doing
   the most work, and the cubic Hamiltonian the least.
4. **The continuation tracks.** At epoch 1500 the tolerance is `1/1499 = 6.67e−4` and all three drifts
   sit at that order (`h1` feasible, `mass` and `h2` over by `3.0e−4` and `1.7e−4`).

The objective rising after epoch 1001 is not an instability: `λ(k)` ramps the `‖r_x‖²` term in from
epoch 1000 (§4), so a new term enters the objective there.

### 9.2 A 1200-epoch KdV run — works mechanically, plateaus infeasible *(verified)*

`type_pde = 1`, hard IC + BC, all three invariants, `GRACE = 0`, 47 s:

| epoch | objective | IC MSE | BC MSE | `mass` drift (μ) | `mom` drift (μ) | `ham` drift (μ) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.28e+00 | 0 | 6.4e−15 | 7.2e−03 (0) | 3.5e−02 (0) | 1.1e−02 (0) |
| 301 | 4.32e−01 | 0 | 1.5e−15 | 2.7e−02 (1.8) | 3.7e−02 (3.0) | 2.5e−02 (0.29) |
| 601 | 1.56e−01 | 0 | 7.1e−16 | 4.4e−03 (2.4) | 9.4e−03 (3.2) | 5.6e−02 (0.45) |
| 1200 | 1.64e−01 | 0 | 6.3e−16 | 2.4e−02 (3.4) | 1.0e−02 (3.4) | 1.4e−02 (0.81) |

**What the ansatz delivers.** `IC MSE ≡ 0` and `BC MSE ~ 6e−16` throughout, and the objective descends
from epoch 1 with **no grace period** — where the previous formulation needed `H_GRACE = 1000` and
even then left the IC term frozen at `0.511` (= mean cos²) for thousands of epochs. The trap is gone.

**What does not converge.** The objective plateaus at `≈ 0.16` from epoch 500 and none of the three
constraints is satisfied at the end (`mass` over by `2.3e−2`, `mom` by `9.6e−3`, `ham` by `4.0e−3`),
with the drifts oscillating rather than settling — `ham` swings `1.3e−2 → 5.6e−2 → 7.2e−2 → 1.4e−2`.

This is the **expected** outcome, and it corroborates `NOTES.md` §5.2 rather than indicating a defect
in the formulation: `mean(r²)` cannot descend much below `0.16` on a 45-point grid for an instance
whose solution carries harmonic index ≈ 91 (§6.7). The constrained program is being solved as well as
this discretisation permits. **KdV needs the resolution fixed before its numbers mean anything;** the
hard-encoding and hierarchy machinery is nonetheless verified to work on it.

A reporting note from this run: the conventional relative drift printed
`|ΔI|/|I⁰| = 6.3e+05` for `mass`, because `|I⁰| = 4.8e−08 > 0` passed an absolute guard. That column
now uses the same *relative* degeneracy test as the scale itself and reports `NaN` there
*(verified)* — the constraint value `drift/S = 2.4e−2` is the meaningful number.

## 10. What is now measurable

The formulation supports a clean ablation that was not previously expressible:

- **encoding** ∈ {soft IC+BC, hard IC+BC} — isolates what the ansatz buys, independent of the duals;
- **constraint set** `𝒥` ∈ {∅, {H}, {mass, quadratic}, {mass, quadratic, H}} — isolates the value of
  the hierarchy against the Hamiltonian alone;
- **reported per epoch**: PDE residual, IC/BC residuals as *diagnostics* (they verify the ansatz
  rather than being trained), per-invariant drift in both scalings, and every multiplier.

Two gaps remain before the central claim is testable, both from `NOTES.md`:

1. **No ground truth** (§5.8). Every current diagnostic is self-referential. CH is well resolved and
   the spectral solver written for §3.3 already produces a reference solution for it — wiring that in
   would give a true `L²` error curve, which is what "constraining `H` improves the solution" actually
   requires.
2. **KdV is under-resolved** (§5.2), now confirmed from a second direction: the dispersive oscillation
   sits at harmonic index ≈ 91 against `K = 8` and a grid Nyquist of 22.

---

## 11. The reference solution *(implemented 2026-09-09)*

§10 lists what is measurable; this is what makes it so. §5.8 of `NOTES.md` was the binding gap: the
residual, the IC/BC mismatch and the invariant drift are all computed from the network's own output,
so none of them can distinguish a good solution from a wrong one that happens to be self-consistent.
The frozen field `u(x,t) = u_0(x)` is the extreme case — it conserves every invariant in §3 exactly.

### 11.1 What is integrated, and why the two solvers differ

Both scripts integrate their *own* equation, with their own signs and normalisations, on a periodic
Fourier grid of `REF_NX` points (right endpoint excluded, so no node is double-counted).

| | equation integrated | scheme | why |
| --- | --- | --- | --- |
| CH | `m_t = −(u m_x + 2 u_x m)`, `u = 𝒜⁻¹m` | plain RK4, transport CFL | no stiff linear term: `𝒜⁻¹ = (1−∂_xx)⁻¹` is smoothing |
| KdV | `u_t = α(u²)_x + ρu_x + νu_xxx` | integrating-factor RK4 | `νu_xxx` is stiff; the linear part is solved exactly |

For KdV, write the equation in Fourier space as `v_t = α(ik)ℱ[u²] + L(k)v` with
`L(k) = i(ρk − νk³)`. The substitution `w = e^{−L τ}v` removes the linear part identically,

```
w_τ = e^{−L τ} · α(ik) ℱ[u²],
```

and `τ` is measured from the start of each step so the phase never grows large.

**The step is set by the dispersive term, not the CFL** *(verified)*. This is the one non-obvious
point. The integrating factor removes the linear *stability* limit but not the accuracy cost of a
large `|L(k)|h` across the RK4 stages, so the nonlinear CFL is the wrong thing to size from. At
`N = 512`, `t_max = 5`, writing `rate = |ν|k³_max + c·k_max`:

| `h · rate` | Hamiltonian drift |
| --- | --- |
| 62 | 3.6e−01 |
| 16 | 9.9e−11 |
| 4 | 3.0e−12 |

Refining the *grid* instead does not help at all: `N = 2048` at `h·rate = 62` still fails, and
`N = 1024` there overflows to `nan`. So `REF_DT_SAFETY = 8.0` bounds `h · rate`, and `REF_NX = 512`
is already sufficient in space. CH sizes its step from the transport CFL as usual.

### 11.2 The solver verifies itself

A wrong reference is worse than none, so each solver reports the drift of **the same invariant
hierarchy the constraints use** — calling the script's own density functions, so the two cannot
disagree — and raises `SystemExit` if the worst exceeds `REF_MAX_DRIFT = 1e-6`. Measured *(verified)*:

| | `mass` | quadratic | Hamiltonian | cost |
| --- | --- | --- | --- | --- |
| CH, `N=256`, 13 substeps/slice | 3.5e−16 | 6.7e−10 | 8.0e−10 | 0.2 s |
| KdV, `N=512`, 1579 substeps/slice | 1.7e−16 | 6.7e−12 | 4.9e−12 | 20.5 s |

The drift is scaled by the density's `L¹` norm, not `|I⁰|`, for the reason given in §7.3: KdV's
`∫cos πx dx` vanishes exactly, and dividing by it would report a perfectly conserved invariant as a
gross failure. This is the same trap as the `EPS_SCALE` guard.

### 11.3 What is reported

`ref_errors` evaluates the network on the reference grid and returns the relative `L²` error,
globally and per time slice. It runs every `REF_EVERY = 25` epochs (`nan` elsewhere, so the CSV
column stays one row per epoch), and appears in the status line, the final summary, the
`ref_rel_l2` CSV column, and two figures:

- `ref_l2` — true error against epoch, with the frozen-field score drawn as a horizontal line;
- `ref_compare` — four panels: reference `u`, network `u_θ`, their difference, and the per-slice
  error. All three fields are drawn as `(x,t)` heatmaps rather than 3-D surfaces, because a
  perspective surface cannot be read quantitatively (the two periodic boundary traces of a
  correct solution appear to diverge in it) and because in an `(x,t)` heatmap the dynamics are
  legible at a glance: **a travelling wave is slanted stripes, the frozen field is stripes parallel
  to the `t` axis, and a collapse fades out.**

### 11.4 The frozen-field baseline is a tripwire

Startup prints the score of `u(x,t) = u_0(x)`: **0.4222** for CH, **1.3067** for KdV. It is not a
target but a diagnostic threshold — a run reporting good invariant drift and a rel. `L²` near this
number has learned no dynamics at all. The final summary says so outright.

### 11.5 First measurements *(verified)*

Applied to the two saved 1000-epoch CH runs, on a 256×101 grid:

| | `mean(r²)` | mass `0→5` | `h1` | `h2` | rel. `L²` | peak travel | `max\|u − u_0\|` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unconstrained | **9.06e−04** | 1.2566 → **0.2128** | 0.2042 → 0.0399 | 0.0942 → 0.0033 | **0.620** | — | 0.234 |
| constrained | 5.54e−03 | → 1.2605 | → 0.2050 | → 0.0949 | **0.422** | **0.00** | **0.0015** |
| frozen `u_0` | 4.93e−02 | exact | exact | exact | 0.4222 | 0.00 | 0 |
| reference | — | 1.2566 | 0.2042 | 0.094248 | 0 | 1.77 | 0.201 |

Two things follow, and neither was visible before.

1. **The unconstrained run collapses; the constrained run freezes.** Mass loses 83% of its value
   unconstrained (and `u` goes negative), which is what an `H` relative error plateauing at ≈ 0.97
   means. Constrained, the invariants hold to a few tenths of a percent — and the solution is the
   frozen field to three digits, matching its `L²` score to four. This is §6.8's degenerate point
   reached in practice. The constraints did remove the collapse; they did not produce dynamics.
2. **The residual is anti-correlated with true error here.** The unconstrained run has a residual
   6× *lower* and is 1.5× *further* from the truth. Selecting runs or hyperparameters by `pde_loss`
   would systematically pick the worse solution. Every claim in §9 was ranked on quantities with
   this property.

The KdV reference makes §5.2 of `NOTES.md` visual rather than arithmetic: it resolves the
Zabusky–Kruskal soliton lattice — roughly eight solitons crossing repeatedly over `t ∈ [0,5]` —
against a network with `N_x = 45` collocation points and `K = 8` Fourier modes. The 20-epoch
control scores 1.29, essentially the frozen-field value.

---

## 12. The quadrature grid, and the degeneracy it caused *(fixed 2026-09-09)*

§11.5 reported that the constrained runs converge to the frozen field `u(x,t) = u_0(x)`, and read
that as §6.8's degenerate point being reached in practice. That reading was incomplete. The frozen
field was not a basin the optimiser fell into — it was **the essentially unique feasible point of
the program as written**, and the cause was the quadrature.

### 12.1 One grid was doing two unrelated jobs

`I_i^h` was integrated on the *collocation* grid. But the two grids answer different questions:

| | sized by | requirement |
| --- | --- | --- |
| collocation grid | parameter count, `N_x = ⌈√(p/fac)⌉` | how many residual equations to impose (§4) |
| quadrature grid | nothing — it inherited `N_x` | resolve the invariant **density** |

The density is the problem. `f_i` is quadratic (`h1`, `mom`) or cubic (`h2`, `ham`) in `(u, u_x)`, so
its Fourier support is several times wider than `u`'s, and it broadens further as the solution
steepens. Measured on the reference solution *(verified)*:

| highest harmonic holding >1e−6 of peak | `t = 0` | `t = t_max` | Nyquist at `N_x = 45` |
| --- | --- | --- | --- |
| CH `h1` density | 4 | **128** | 22 |
| KdV `mom` density | 2 | **~99** | 22 |

At `t = 0` everything is resolved — `u_0` is band-limited to harmonic 2, so the densities reach 4 or
6 and *every* rule is exact to `1e-16`. By `t = t_max` about 2.4% of the CH `h1` density's energy
sits above the grid's Nyquist and is aliased.

### 12.2 What the quadrature reported for the exact solution *(verified)*

Push the **exact** reference solution, with its exact spectral derivative, through the scripts' own
`Q`. The true drift is `~1e-10` (§11.2), so everything below is pure quadrature error:

| `NX_QUAD` | 45 | 89 | 181 | **361** | 721 |
| --- | --- | --- | --- | --- | --- |
| CH `h1` | 5.19e−02 | 1.05e−02 | 7.53e−04 | **4.00e−05** | — |
| CH `h2` | 5.99e−02 | 1.23e−02 | 8.14e−04 | **4.32e−05** | — |
| KdV `mom` | 2.12e−01 | — | 8.04e−04 | **7.40e−07** | 6.7e−12 |
| KdV `ham` | **1.09e+00** | — | 1.09e−02 | **4.07e−05** | 1.6e−11 |

At `N_x = 45` the KdV quadrature reports **109% drift in the Hamiltonian of the exact solution**.
The tolerance schedule (§7.2) meanwhile tightens `ε_i` to `1e-4`.

This is not a choice of rule. At `N_x = 45`, Boole gives `h1 = 5.19e-2`, the periodic trapezoid
`4.11e-2`, and `quad_rect` as written `4.28e-2`. Nothing on 45 points resolves harmonic 128.

### 12.3 Why the artefact selects the frozen field exactly

This is the part worth keeping. The constraint bounds a **difference**, `|I_i^h(t) − I_i⁰|`, and the
aliasing error is **time-dependent** — identically zero at `t = 0`, order `5e-2` at `t = t_max`.
Write the measured value as the truth plus the rule's error:

```
I^h(t) = I[u(·,t)] + e(t),        so       I^h(t) − I^h(0) = 0 + (e(t) − e(0))
```

for any exactly conserving `u`. The constraint therefore sees `e(t) − e(0)`, and there is exactly
one way to make that vanish identically: `e` must not depend on `t`, which for a rule that depends
on `u` only through its samples means **`u` must not depend on `t`**. Hence `u = u_0`, which is
feasible to machine precision, while every genuinely evolving field — the true solution included —
violates the constraint by two orders of magnitude. Training converged correctly.

### 12.4 Three explanations ruled out *(verified)*

- **Not the optimiser.** Fit `u_θ` to the reference by plain least squares — no PDE, no constraints.
  It reaches rel `L²` `0.0032`, and still violates `h1` by 60×. Its drift does not improve as the
  fit improves: `2.42e-2` at rel `L²` `0.0061`, `2.45e-2` at `0.0032`. It has plateaued on something
  that is not the fit quality.
- **Not the network.** The exact solution through the same quadrature scores slightly *worse*
  (`5.19e-2` against the network's `2.45e-2`). The network was never the weak link.
- **Not `penalty = 0`** (§5.3 of `NOTES.md`). The frozen field is *feasible*, so the augmentation
  term `(ρ/2)‖max(0, c)‖²` is identically zero there. It cannot push away from a feasible point.
  Enabling it makes the ALM converge to the frozen field more reliably, not less.

### 12.5 The fix

`NX_QUAD = 361` in both scripts, a dedicated x-grid at the same collocation times, with `I_i⁰` and
`S_i` moved onto it so the `t = 0` cancellation of §4 is preserved. `(NX_QUAD − 1) ≡ 0 (mod 4)` for
the same reason `N_x` is rounded (§5.1). Overhead is **14%** in wall-clock: 16245 quadrature points
against 2025 collocation points, but the invariants need only `∂_x u` where the residual needs third
and mixed derivatives.

Plus a startup guard, `quadrature_floor`, in the spirit of §11.2 — the reference solution is pushed
through the script's own `Q` and the run **aborts** if any `_EPS_FLOOR` sits below the resulting
floor, naming the invariant and the two numbers. Verified to fire at `NX_QUAD ∈ {45, 181}` and pass
at 361. `ε` is now bounded below by a *measured* quantity rather than by an arbitrary constant.

### 12.6 Necessary, but not sufficient: there are two floors *(verified)*

The mechanism of §12.3 depends only on the measured invariant carrying a **`t`-dependent error**. It
says nothing about the source. Quadrature aliasing was one source; the network's own `∂_x u` error is
another, and the fix does not touch it. Fitting `u_θ` to the reference on the decoupled grid:

| rel `L²` of the fit | `mass` | `h1` | `h2` |
| --- | --- | --- | --- |
| 0.0045 | 1.23e−03 | 3.22e−02 | 4.48e−02 |
| 0.0029 | 8.67e−04 | 2.37e−02 | 3.33e−02 |
| **0.0023** | 1.04e−03 | **1.88e−02** | 2.63e−02 |

against a quadrature floor of `4.0e-05`. So the binding floor is now the network, not the rule: the
invariants depend on `u_x`, and a network's derivative error far exceeds its value error, so a 0.2%
`L²` fit still misconserves `h1` by 2%. The decoupling did change its character — the drift now
*improves* with the fit, where on the coupled grid it was flat at `2.4e-2` — but `1.88e-02` remains
20× above the `1e-3` demanded at 1500 epochs. The frozen field is therefore still uniquely feasible,
and a 1500-epoch run still lands on it (rel `L²` `0.4215` against the frozen `0.4222`, duals climbing
to 6.3/1.5/1.3 with all three constraints violated).

Setting the floor above the measured value breaks the degeneracy: with `_EPS_FLOOR = 5e-2` every
constraint is satisfied with duals ≈ 0 and the run reaches rel `L²` **0.3999**, better than frozen
for the first time. The remaining gap to the representable `0.0023` is ordinary PINN trainability,
not a degeneracy — the constraint has stopped selecting the wrong answer.

The `quadrature_floor` guard checks only the quadrature term, so it passes at `_EPS_FLOOR = 1e-4`
and does not catch this. Covering it means fitting the reference at startup (~60 s).

### 12.7 What this invalidates

Every invariant-drift number in §9 and §11.5, and the §9.1 multiplier ordering, was measured while
the constraint was reading its own aliasing error. The qualitative claims of §6 (hard IC/BC hold
identically) are unaffected — they never involved a quadrature. The degeneracy-order result of §6.8
is unaffected as mathematics, but it is no longer the explanation for what the runs did.
