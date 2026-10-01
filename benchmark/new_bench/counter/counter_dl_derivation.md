# Closed-form derivation for `counter_dl.ipynb`

Every number the notebook prints is derived here in closed form. Section 9 lists each claim
against its measured value.

## 1. Problem

Model, one hidden ReLU unit with a linear read-out:

$$\hat y(x;\theta) = v\,\mathrm{relu}(wx+b),\qquad \theta=(w,b,v)\in\mathbb R^3 .$$

Three design points, $n = 1000$ in total:

| $x$ | $y$ | count | weight | group |
|---|---|---|---|---|
| $0$ | $0$ | 495 | $0.495$ | A |
| $2$ | $0$ | 495 | $0.495$ | A |
| $1$ | $1$ | 10 | $0.01$ | B |

Write $s_0=b$, $s_2=2w+b$, $s_1=w+b$ for the three pre-activations and $\hat y_j = v\,\mathrm{relu}(s_j)$.

$$f(\theta)=0.495\,\hat y_0^{\,2}+0.495\,\hat y_2^{\,2}+0.01(\hat y_1-1)^2,
\qquad g(\theta)=(\hat y_1-1)^2-\tfrac14\le 0 .$$

## 2. The dead cone, and why one unit cannot make a bump

The pre-activation $s(x)=wx+b$ is **affine in $x$**, and the minority point $x=1$ is the midpoint
of the two majority points $x\in\{0,2\}$. Hence

$$s_1=\tfrac12(s_0+s_2). \tag{2.1}$$

So $s_0\le 0$ and $s_2\le0$ force $s_1\le0$: a single unit cannot be off at $x=0$ and $x=2$ while
being on at $x=1$. Fitting group A necessarily switches group B off. Define

$$D=\{\theta:\ b\le 0,\ 2w+b\le 0\}.$$

$D$ is the intersection of two half-planes through the origin, hence a closed convex cone with
apex at the origin; its boundary rays point along $(-1,0)$ and $(1,-2)/\sqrt5$, so its opening
angle is

$$\arccos\!\left(\tfrac{-1}{\sqrt5}\right)=116.565^\circ, \tag{2.2}$$

i.e. $32.4\%$ of all directions in the $(w,b)$ plane. In $\mathbb R^3$ the trap is
$\operatorname{int}D\times\mathbb R_v$, an open set of positive measure — not a measure-zero kink.

On $\operatorname{int}D$ all three relus are strictly inside their inactive branch, so
$\hat y_0=\hat y_1=\hat y_2=0$ *identically in a neighbourhood*, and

$$f\equiv 0.01,\qquad g\equiv \tfrac34,\qquad \nabla f\equiv 0,\qquad \nabla g\equiv 0. \tag{2.3}$$

**The dual goes blind on a strictly larger set.** $g$ depends on $\theta$ only through
$\hat y_1=v\,\mathrm{relu}(w+b)$, so $\nabla g=0$ on the whole half-plane $\{w+b<0\}\supsetneq
\operatorname{int}D$, where $g\equiv\frac34>0$. This is what produces the two-stage failure of
Section 6: the constraint can be invisible while the objective still moves.

## 3. Every dual method in `dual_optim` stalls on $D$

Each method forms a scalar surrogate whose primal gradient is

$$\nabla_\theta \mathcal L=\nabla f+c(\lambda,\rho,g)\,\nabla g ,$$

with $c=\lambda+\rho[g]_+$ (ALM, quadratic), $c=\max(0,\lambda+\rho g)$ (ALM, HPR), $c=$ the
penalty-barrier derivative (PBM), and for SSG the surrogate is $f$ or $\max_j c_j$, so the
gradient is $\nabla f$ or $\nabla g$. In every case the direction lies in
$\mathrm{span}\{\nabla f,\nabla g\}$, and by (2.3) both vanish on $\operatorname{int}D$. Therefore

$$\nabla_\theta\mathcal L\equiv 0 \ \text{ on } \operatorname{int}D,\ \text{ for every } (\lambda,\rho), \tag{3.1}$$

so the primal step is zero whatever the dual does, and the stall is independent of the method,
of $\lambda_0$ and of $\rho$. Meanwhile the dual update sees only constraint *values*, and
$g=\frac34$ is constant, so for ALM with dual step $\gamma$

$$\lambda_k=\lambda_0+\tfrac34\gamma k. \tag{3.2}$$

At $\gamma=0.1$ this gives $\lambda_k=0.075k$; after 400 iterations $\lambda_{399}=29.925$.

Note what is *not* violated: $\|\nabla\mathcal L\|\to0$ and the primal iterate converges. Only
asymptotic feasibility fails.

## 4. Neither $f$ nor $g$ is weakly convex

**Lemma.** If $h:\mathbb R\to\mathbb R$ is $\rho$-weakly convex then $h+\frac\rho2 t^2$ is convex,
so its derivative $h'+\rho t$ is nondecreasing. Adding $\rho t$ is continuous, so a jump
discontinuity of $h'$ survives unchanged; monotonicity therefore requires every jump to be
$\ge 0$. A **downward** slope jump rules out every finite $\rho$. Weak convexity also passes to
lines: if $g$ is $\rho$-weakly convex then $t\mapsto g(\theta_0+td)$ is $\rho\|d\|^2$-weakly
convex, so one bad line suffices.

Take the slice $w=0$, where (2.1) collapses all three pre-activations onto $b$:

$$F(b):=f(0,b,1)=\begin{cases}0.01,& b\le0\\ b^2-0.02b+0.01,& b>0\end{cases}
\qquad
G(b):=g(0,b,1)=\begin{cases}0.75,& b\le0\\ b^2-2b+0.75,& b>0\end{cases} \tag{4.1}$$

$$F'(0^-)=0,\quad F'(0^+)=-0.02,\qquad G'(0^-)=0,\quad G'(0^+)=-2 .$$

Both jump **downward** at $b=0$, so neither is $\rho$-weakly convex for any $\rho$. These are the
two curves the notebook plots, and they are the exact analogue of the concave kinks at $x=2$ and
$x=5$ in `counter.ipynb`.

**Why weak convexity would have mattered.** If $g$ were $\rho$-weakly convex then from
$\nabla g(\theta_0)=0$,

$$g(z)\ \ge\ g(\theta_0)-\tfrac\rho2\|z-\theta_0\|^2 \quad\Longrightarrow\quad
\|z-\theta_0\|\ \ge\ \sqrt{2g(\theta_0)/\rho}$$

for every feasible $z$. Here $\operatorname{dist}(D,\{w+b\ge\frac12\})=\frac{1}{2\sqrt2}$
(attained as the cone approaches its apex), so $d^2=\frac18$ and any weakly convex $g$ with this
trap would need

$$\rho\ \ge\ \frac{2\cdot\frac34}{\frac18}\ =\ 12 . \tag{4.2}$$

For convex $g$ ($\rho=0$) the trap is impossible outright: $\nabla g(\theta_0)=0$ would make
$\theta_0$ a global minimiser, so $g(\theta_0)>0$ would mean the *problem* is infeasible.

## 5. The two optima

Since $(w,b)\mapsto(s_0,s_2)$ is a bijection ($b=s_0$, $w=\frac12(s_2-s_0)$) and $s_1$ is pinned
by (2.1), minimise in $(s_0,s_2)$ at $v=1$.

**Constrained.** Feasibility is $\hat y_1\in[\frac12,\frac32]$, so $m:=s_1\ge\frac12>0$. For fixed
$m$, minimising $\mathrm{relu}(s_0)^2+\mathrm{relu}(s_2)^2$ subject to $s_0+s_2=2m$ gives
$s_0=s_2=m$: writing $s_{0,2}=m\mp\delta$ gives $2m^2+2\delta^2$, and moving either below $0$ only
pushes the other past $2m$, costing $4m^2>2m^2$. So $w=0$, $b=m$, and the problem reduces to
minimising $F(m)=m^2-0.02m+0.01$ over $m\in[\frac12,\frac32]$. $F'=2m-0.02>0$ there, so

$$\boxed{\ \theta^\star=(0,\tfrac12,1),\qquad \hat y\equiv\tfrac12,\qquad f^\star=\tfrac14,\ } \tag{5.1}$$

with $g=0$ active. The unit predicts $\frac12$ for *everyone*: by (2.1) the only way one unit can
lift the middle point is to lift both ends with it. That is the price of the constraint, and it is
$25\times$ the unconstrained optimum.

**Multiplier.** At $\theta^\star$ all $s_j=\frac12>0$, so everything is smooth and
$\nabla f=(0.98,0.98,0.49)$, $\nabla g=(-1,-1,-0.5)$. Then $\nabla f+\lambda\nabla g=0$ in all
three coordinates at

$$\lambda^\star=\tfrac{49}{50}=0.98\ \ge 0, \tag{5.2}$$

so KKT holds and the problem is entirely well posed.

**Unconstrained.** In the all-active region, $\partial f/\partial s_0=0.99s_0+0.01(m-1)$ and
likewise for $s_2$, giving $s_0=s_2=m$ and $0.99m+0.01(m-1)=0$, i.e.

$$m=0.01:\qquad (w,b)=(0,\tfrac1{100}),\qquad f_{\min}=0.0099 . \tag{5.3}$$

Letting $s_0\le0$ instead forces $s_2\ge 2m$ and costs $1.99m^2-0.02m\ge-5.03\cdot10^{-5}$, i.e.
$f\ge0.00994975>0.0099$. So (5.3) is the global minimum.

> **Correction to the notebook.** The notebook asserts that the dead cone *is* the objective's
> global minimiser set. That is wrong: the cone gives $f=0.01$, while the global minimum is
> $f=0.0099$ at $(0,0.01)$, just outside it. The cone is a **local** minimiser set (every point of
> $\operatorname{int}D$ is a local minimum because $f$ is locally constant there), at $1.0101\times$
> the global value. The conclusion is unaffected, and arguably sharpened: the trap costs the
> objective $1\%$, which is why nothing in the loss curve flags it, while costing feasibility
> permanently. Plain SGD on $f$ alone does land in the cone from a natural initialisation.

## 6. Dynamics from the initialisation

At $\theta_0=(0.1,0.8,1)$: $s_0=0.8$, $s_2=1.0$, $s_1=0.9$, all active, so $f=0.8119$ and
$g=-0.24<0$. The constraint is *satisfied*, so $\lambda_1=[\lambda_0+\gamma g]_+=0$ and
$[g]_+=0$: under both augmentations the surrogate is **exactly $f$** at step 0 — the dual has no
information yet. There

$$\nabla f(\theta_0)=(1.978,\ 1.780,\ 1.6218). \tag{6.1}$$

$v$ does not enter any $s_j$, so one step $\eta$ sends $w\mapsto0.1-1.978\eta$,
$b\mapsto0.8-1.780\eta$ whether or not $v$ is trained, and each design point switches off at its
own threshold:

$$\begin{aligned}
s_2\le0 &\iff 1.0-5.736\,\eta\le0 &&\iff \eta\ge \tfrac{125}{717}=0.17434\\
s_1\le0 &\iff 0.9-3.758\,\eta\le0 &&\iff \eta\ge \tfrac{450}{1879}=0.23949 \quad(\nabla g=0:\text{ dual blind})\\
s_0\le0 &\iff 0.8-1.780\,\eta\le0 &&\iff \eta\ge \tfrac{40}{89}=0.44944 \quad(\nabla f=0\text{ too: full trap})
\end{aligned} \tag{6.2}$$

**Regime $\eta\ge0.44944$.** The iterate lands in $D$ at step 1 and freezes at
$(0.1-1.978\eta,\ 0.8-1.780\eta)$, with $\|\nabla\mathcal L\|=0$ exactly ever after.

**Regime $0.23949\le\eta<0.44944$.** After step 1 only $x=0$ is active, so
$f=0.495\,v^2b^2+0.01$ and $g\equiv\frac34$. Since $s_0=b$ does not involve $w$,

$$\frac{\partial f}{\partial w}=0,\qquad \frac{\partial f}{\partial b}=0.99\,v^2 b,\qquad \nabla g=0 .$$

So $w$ is **frozen** at $w_1=0.1-1.978\eta$ and, with $v$ fixed at 1, $b$ decays geometrically:

$$b_{k+1}=\bigl(1-0.99\,\eta\bigr)b_k,\qquad b_k=b_1(1-0.99\eta)^{k-1},\quad b_1=0.8-1.780\eta. \tag{6.3}$$

For $\eta<1/0.99$ the factor lies in $(0,1)$, so $b\to0^+$ but **never reaches $0$**: in exact
arithmetic the iterate converges to the cone's *boundary* $(w_1,0)$ with $\|\nabla\mathcal L\|\to0$
without ever being $0$. It is nonetheless just as stuck, because the only surviving gradient
component is the objective's and it points further into the cone. In float32 the gradient flushes
to exactly zero once $b\lesssim3.7\cdot10^{-23}=\sqrt{\text{smallest subnormal}}$, i.e. once the
per-point squared error underflows (measured: iteration 145 at $\eta=0.30$) — a floating-point
artifact, not part of the derivation.

**Regime $\eta\le0.20$.** The constraint pressure wins and ALM converges to (5.1).

## 7. Why escalating the penalty walks into the trap

At a point where all three units are active and $v=1$,

$$H_f=\begin{pmatrix}3.98&2.00\\2.00&2.00\end{pmatrix},\qquad
\nabla g=2(w+b-1)\,(1,1)^{\!\top},$$

and the ALM surrogate contributes $\nabla^2\bigl[\lambda g+\tfrac\rho2[g]_+^2\bigr]\approx
(2\lambda+\rho)\,(1,1)(1,1)^{\!\top}$ near the boundary. The rank-one term has eigenvalues
$2(2\lambda+\rho)$ and $0$, so

$$L_{\max}(\mathcal L)\ \approx\ 2\rho+4\lambda+\mathcal O(1),$$

and gradient descent is stable only while $\eta<2/L_{\max}$, i.e.

$$\boxed{\ \eta\,\rho\ \lesssim\ 1\ } \tag{7.1}$$

A penalty schedule that grows $\rho$ at fixed $\eta$ therefore *must* eventually violate the
descent condition. On a smooth or weakly convex problem the consequence is oscillation and then
divergence; here the overshoot lands in $D$, where it stops. The constant in (7.1) is only
indicative because the penalty curvature is one-sided at an active constraint; measured at
$\eta=0.10$, $\rho=10$ still converges and $\rho=20$ traps, i.e. the threshold sits at
$\eta\rho\in(1,2)$.

The *multiplier* is not the culprit: by (5.2) it tracks $\lambda^\star=0.98$ and stays bounded
while the run is healthy; its divergence (3.2) is a consequence of the trap, not a cause.

## 8. What escapes

- **Exact prox / trust region.** $\arg\min_\theta\{\mathcal L(\theta)+\frac1{2\eta}\|\theta-\theta_k\|^2\}$
  leaves $D$ once $\eta$ exceeds a finite threshold, because $\mathcal L$ is not globally constant.
  A penalty-barrier method that solves this subproblem with an inner first-order solver started at
  $\theta_k$ does **not**, since the inner solver also sees (3.1).
- **Randomised smoothing / Goldstein $\varepsilon$-subdifferential**: works iff $\varepsilon$
  exceeds the distance to the kink.
- **Smooth activation**: `softplus` removes the flat branch and converges to (5.1).
- **Leaky activation**: insufficient here. The leak lets $\hat y_1<0$, which inflates
  $(\hat y_1-1)^2$ and hence the penalty; at $\eta=0.30$ the run diverges to NaN instead of
  stalling.
- **Dead-unit reinitialisation**: leaves $D$, but at a step size that caused the death the run
  re-enters it, giving a cycle; it must be paired with fixing the overshoot.
- **Not**: a larger $\lambda$ or $\rho$ (by (3.1)), minibatch noise (the gradient on $D$ is exactly
  zero with *zero variance*, so no saddle-escape argument applies — $D$ is a flat local minimum of
  $\mathcal L$, not a strict saddle), momentum (decays to zero), or Adam ($0/\sqrt0$).

## 9. Derived versus measured

| quantity | closed form | value | notebook |
|---|---|---|---|
| $f$ on $D$ | $1/100$ | $0.01$ | `0.010000` |
| $g$ on $D$ | $3/4$ | $0.75$ | `+0.7500` |
| global $\min f$ | at $(0,\tfrac1{100})$ | $0.0099$ | grid search ✓ |
| constrained opt | $(0,\tfrac12,1)$, $f^\star=\tfrac14$ | $0.25$ | `0.250000` |
| multiplier | $\lambda^\star=\tfrac{49}{50}$ | $0.98$ | `0.9798` |
| $\nabla f(\theta_0)$ | (6.1) | $(1.978,1.780,1.6218)$ | ✓ |
| $\|\nabla\mathcal L(\theta_0)\|$, $v$ frozen | $\sqrt{7.080884}$ | $2.66099$ | `2.661e+00` |
| $s_2$ off | $125/717$ | $0.17434$ | — |
| $s_1$ off (dual blind) | $450/1879$ | $0.23949$ | blind@1 from $\eta=0.25$ |
| $s_0$ off (full trap) | $40/89$ | $0.44944$ | exactly0@1 from $\eta=0.45$ |
| decay ratio, $\eta=0.30$ | $1-0.99\eta$ | $0.703$ | $0.266\to0.187\to0.1315$ |
| frozen $w$, $\eta=0.25/0.30/0.35/0.40$ | $0.1-1.978\eta$ | $-0.3945/-0.4934/-0.5923/-0.6912$ | ✓ all four |
| landing point, $\eta=0.45/0.50$ | $(0.1-1.978\eta,\ 0.8-1.780\eta)$ | $(-0.7901,-0.001)$, $(-0.889,-0.09)$ | ✓ |
| multiplier growth | $\lambda_k=0.075k$ | $\lambda_{399}=29.925$ | `29.9251` |
| cone opening angle | $\arccos(-1/\sqrt5)$ | $116.565^\circ$ | — |
| $\operatorname{dist}(D,\text{feasible})$ | $1/(2\sqrt2)$ | $0.35355$ | — |
| weak-convexity modulus needed | $2g/d^2$ | $\ge 12$ | — |
