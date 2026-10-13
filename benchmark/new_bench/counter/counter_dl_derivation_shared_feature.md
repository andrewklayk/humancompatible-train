# Closed-form derivation for the shared-feature counterexample

A single design point, `x = 1`, carries **both** labels: the two groups are separated only by
`y`, not by `x`. This document records the closed forms for the resulting trap.

Numbers below are for ALM with `penalty = 1`, dual step `0.1`. **Note:** commit `5812779` made
`hpr` the default augmentation whenever `penalty > 0`; where the two augmentations differ, both
are given.

## 1. Problem

$$\hat y(x;\theta)=v\,\mathrm{relu}(wx+b),\qquad \theta=(w,b,v)\in\mathbb R^3 .$$

| $x$ | $y$ | count | weight | group |
|---|---|---|---|---|
| $1$ | $0$ | 990 | $0.99$ | A |
| $1$ | $1$ | 10 | $0.01$ | B |

Every input coincides, so there is a **single** pre-activation $s=w+b$ and a single prediction
$\hat y=v\,\mathrm{relu}(s)$ shared by both groups. The model therefore cannot tell the groups
apart at all; the tension is pure label conflict at one point.

$$f(\theta)=0.99\,\hat y^{\,2}+0.01(\hat y-1)^2,\qquad
g(\theta)=(\hat y-1)^2-\tfrac14\le 0 .$$

Both are functions of the *scalar* $\hat y$ alone:

$$f=\hat y^{\,2}-0.02\,\hat y+0.01,\qquad g=\hat y^{\,2}-2\hat y+0.75 . \tag{1.1}$$

## 2. The dead half-plane

$\hat y$ enters only through $\mathrm{relu}(s)$, so on the open half-plane

$$D=\{\theta:\ w+b<0\}$$

we have $\hat y\equiv0$ identically in a neighbourhood, hence

$$f\equiv0.01,\qquad g\equiv\tfrac34,\qquad \nabla f\equiv0,\qquad \nabla g\equiv0. \tag{2.1}$$

The trap is a **half-plane** — half of all directions in $(w,b)$ — and $\nabla f$ and $\nabla g$
vanish on the *same* set: there is no intermediate regime where one keeps moving while the other
goes blind. The unit dies and everything freezes in the same instant.

$\nabla_\theta\hat y=(v\,\mathrm{relu}'(s)x,\ v\,\mathrm{relu}'(s),\ \mathrm{relu}(s))=(0,0,0)$ on
$D$, so (2.1) holds for the $v$-coordinate too — the trap is $D\times\mathbb R_v$, open in
$\mathbb R^3$.

## 3. Every dual method stalls on $D$

Each method in `dual_optim` forms a surrogate with
$\nabla_\theta\mathcal L=\nabla f+c(\lambda,\rho,g)\,\nabla g$, where
$c=\lambda+\rho[g]_+$ (quadratic), $c=\max(0,\lambda+\rho g)$ (HPR), the barrier derivative (PBM),
or, for SSG, the surrogate is $f$ or $\max_j c_j$. In all cases the direction lies in
$\mathrm{span}\{\nabla f,\nabla g\}$, so by (2.1)

$$\nabla_\theta\mathcal L\equiv0\ \text{ on } D,\ \text{ for every }(\lambda,\rho). \tag{3.1}$$

The dual sees only constraint *values*, and $g\equiv\frac34$ is constant on $D$, so

$$\lambda_k=\lambda_0+\tfrac34\gamma k\ \xrightarrow[k\to\infty]{}\ \infty, \tag{3.2}$$

which at $\gamma=0.1$ gives $\lambda_k=0.075k$ and $\lambda_{399}=29.925$. So
$\|\nabla\mathcal L\|\to0$ and the primal converges, while feasibility fails and the multiplier
diverges.

## 4. Neither $f$ nor $g$ is weakly convex

With $v=1$, $\hat y=\mathrm{relu}(s)$ and (1.1) gives

$$F(s)=\begin{cases}0.01,& s\le0\\ s^2-0.02s+0.01,& s>0\end{cases}\qquad
G(s)=\begin{cases}0.75,& s\le0\\ s^2-2s+0.75,& s>0\end{cases} \tag{4.1}$$

$$F'(0^-)=0,\ F'(0^+)=-0.02;\qquad G'(0^-)=0,\ G'(0^+)=-2 .$$

Both slopes jump **downward** at $s=0$. Adding $\frac\rho2 s^2$ perturbs a derivative continuously,
so a jump survives unchanged, and convexity needs a nondecreasing derivative: no finite $\rho$
works, for either function. (Weak convexity passes to lines, so one bad line settles it in
$\mathbb R^3$.)

If $g$ were $\rho$-weakly convex, $\nabla g(\theta_0)=0$ would give
$\|z-\theta_0\|\ge\sqrt{2g(\theta_0)/\rho}$ for every feasible $z$. Here
$\operatorname{dist}(D,\{s\ge\frac12\})=\frac1{2\sqrt2}$, so such a $g$ would need
$\rho\ge2(\frac34)/\frac18=12$. For convex $g$ the trap is impossible outright.

## 5. The two optima

By (1.1) everything reduces to a scalar problem in $\hat y$.

**Unconstrained.** $f'(\hat y)=2\hat y-0.02=0$ gives

$$\hat y_{\min}=0.01,\qquad f_{\min}=0.0099 , \tag{5.1}$$

against $f=0.01$ on $D$. The dead half-plane is therefore a **local** minimiser set at
$1.0101\times$ the global value — a $1\%$ gap. That is the whole reason the failure is invisible:
the objective barely notices the trap, while feasibility is destroyed permanently.

**Constrained.** $g\le0\iff\hat y\in[\frac12,\frac32]$, and $f'(\hat y)>0$ there, so

$$\boxed{\ \hat y^\star=\tfrac12,\qquad f^\star=\tfrac14,\qquad
\lambda^\star=\tfrac{f'(\hat y^\star)}{-g'(\hat y^\star)}=\tfrac{0.98}{1}=\tfrac{49}{50}\ } \tag{5.2}$$

with $g=0$ active and $\lambda^\star\ge0$: KKT holds, the problem is well posed, and the constraint
costs $25\times$ the unconstrained optimum.

**Degeneracy.** Only $\hat y$ is pinned, so the optimal set is the surface
$\{v\,\mathrm{relu}(w+b)=\frac12\}$ — at $v=1$, the whole line $w+b=\frac12$. This is why the
measured $s^\star$ depends on the step size: the run lands somewhere on that surface, not at a
fixed point of the plane.

## 6. Dynamics from the initialisation

At $\theta_0=(1,-0.1,1)$: $s=0.9$, $\hat y=0.9$, so $f=0.802$ and $g=-0.24<0$. The constraint is
*satisfied*, so $\lambda_1=[\lambda_0+\gamma g]_+=0$ and $c=0$ under both augmentations: the
surrogate is **exactly $f$** at step 0 — the dual has no information yet. With
$f'(\hat y)=2\hat y-0.02=1.78$ and $\nabla_\theta\hat y=(v,v,\mathrm{relu}(s))=(1,1,0.9)$,

$$\nabla f(\theta_0)=1.78\,(1,\ 1,\ 0.9)=(1.78,\ 1.78,\ 1.602), \tag{6.1}$$

with norm $1.78\sqrt2=2.5173$ ($v$ frozen) or $1.78\sqrt{2.81}=2.98382$ ($v$ trainable).

Since $s=w+b$ and both partials of $\hat y$ w.r.t. $w,b$ equal $v$, one step moves

$$\Delta s=-2\eta\,v_0\,\frac{\partial\mathcal L}{\partial\hat y}=-3.56\,\eta \tag{6.2}$$

(at $\lambda_0=0$, $v_0=1$), so the unit dies **on the first step** iff

$$s_1=0.9-3.56\,\eta\le0\iff \eta\ \ge\ \tfrac{45}{178}=0.252809 . \tag{6.3}$$

This is a single threshold: there is no blind-but-moving regime.

**Seeded multiplier.** With $\lambda_0>0$ the constraint contributes at step 0. For $\lambda_0=1$,
$\gamma=0.1$, $\rho=1$: the dual update gives $\lambda_1=0.976$ under both augmentations, and the
surrogate's coefficient on $\nabla g$ is

$$c=\max(0,\lambda_1+\rho g)=0.736\ \text{(HPR)},\qquad c=\lambda_1=0.976\ \text{(quadratic)} .$$

With $g'(\hat y_0)=2(\hat y_0-1)=-0.2$, (6.2) becomes $\Delta s=-2\eta(1.78+c\cdot(-0.2))$, so at
$\eta=0.3$

$$s_1=-0.07968\ \text{(HPR)},\qquad s_1=-0.05088\ \text{(quadratic)}, \tag{6.4}$$

both negative: seeding the multiplier does not save the run, it only changes where in $D$ it lands.

## 7. The second route in, and why $v$ closes it

For $\eta$ below (6.3) the unit survives step 0, and the question is whether the fixed point
$\hat y^\star=\frac12$ is stable. With $v$ frozen the iteration in $\hat y=s$ is

$$s_{k+1}=s_k-2\eta\,\mathcal L'(s_k),\qquad\text{multiplier } 1-2\eta\,\mathcal L'' ,$$

so stability requires $\eta\,\mathcal L''<1$. Under **HPR** the coefficient
$c=\max(0,\lambda+\rho g)$ is differentiable through $g=0$, so the same expression holds on both
sides:

$$\mathcal L''=f''+\rho\,(g')^2+\lambda\,g''=2+\rho\cdot1+2\lambda^\star=3.96+\rho, \tag{7.1}$$

$$\eta^\star=\frac{1}{3.96+\rho}\ \overset{\rho=1}{=}\ 0.20161 . \tag{7.2}$$

Measured: converges at $\eta=0.2016$, stalls at $\eta=0.205$. Under the **quadratic** augmentation
the linear term has constant coefficient $\lambda$, so the penalty curvature acts only on the
violated side, $\mathcal L''_+=4.96$ and $\mathcal L''_-=3.96$; a period-2 orbit alternating across
the boundary loses stability when $(2\eta\mathcal L''_+-1)(2\eta\mathcal L''_--1)=1$, i.e.

$$\eta^\star=\frac{\mathcal L''_++\mathcal L''_-}{2\,\mathcal L''_+\mathcal L''_-}=\frac{8.92}{2(4.96)(3.96)}=0.22707, \tag{7.3}$$

against measured convergence at $0.22$ and stalling at $0.227$. So below (6.3) there is a second,
slower route into $D$: the fixed point destabilises, the iterate oscillates with growing amplitude,
and one downswing crosses $s=0$.

**With $v$ trainable that route disappears** — measured, the boundary is exactly (6.3). The reason
is that $v$ is a free step-size controller. The $\hat y$-space step carries the factor
$\|\nabla_\theta\hat y\|^2=2v^2+s^2$, and by Section 5 the optimum only pins $\hat y=v\,s=\frac12$,
so the run is free to settle at whatever $v$ makes

$$\eta\,\mathcal L''\,(2v^2+s^2)<2 \tag{7.4}$$

hold. It does: at $\eta=0.2$ it lands at $v=0.790$, $s=0.6328$ ($2v^2+s^2=1.649$, LHS $=1.636$),
and at $\eta=0.2528$ at $v=0.696$, $s=0.7180$ ($2v^2+s^2=1.485$, LHS $=1.862$) — larger steps are
met with a smaller read-out weight. Only the one-step overshoot (6.3), which happens before $v$ can
adapt, gets through.

## 8. Derived versus measured

| quantity | closed form | value | measured |
|---|---|---|---|
| $f$ on $D$ | $1/100$ | $0.01$ | `0.010000` |
| $g$ on $D$ | $3/4$ | $0.75$ | `+0.750000` |
| global $\min f$ | at $\hat y=1/100$ | $0.0099$ | `0.009900` |
| constrained opt | $\hat y^\star=\frac12$, $f^\star=\frac14$ | $0.25$ | `0.250000` |
| multiplier | $\lambda^\star=\frac{49}{50}$ | $0.98$ | $0.98$ in all 3 coords |
| $f(\theta_0)$, $g(\theta_0)$ | — | $0.802$, $-0.24$ | `0.802000`, `-0.240000` |
| $\nabla f(\theta_0)$ | $1.78(1,1,0.9)$ | $(1.78,1.78,1.602)$ | ✓ |
| $\|\nabla f(\theta_0)\|$ | $1.78\sqrt2$ / $1.78\sqrt{2.81}$ | $2.5173$ / $2.98382$ | `2.517`, `2.984` |
| one-step death | $45/178$ | $0.252809$ | conv $\le0.2528$, stall $\ge0.255$ ($v$ trainable) |
| landing, $\eta=0.3$, $\lambda_0=0$ | $0.9-3.56\eta$ | $-0.168$ | `-0.168000` |
| landing, $\eta=0.3$, $\lambda_0=1$ | (6.4) HPR / quad | $-0.07968$ / $-0.05088$ | ✓ both |
| stability, $v$ frozen, HPR | $1/(3.96+\rho)$ | $0.20161$ | conv $0.2016$, stall $0.205$ |
| stability, $v$ frozen, quadratic | (7.3) | $0.22707$ | conv $0.22$, stall $0.227$ |
| multiplier growth | $\lambda_k=0.075k$ | $\lambda_{399}=29.925$ | `29.9251` |
| $\operatorname{dist}(D,\text{feasible})$ | $1/(2\sqrt2)$ | $0.35355$ | — |
| weak-convexity modulus needed | $2g/d^2$ | $\ge12$ | — |

The $\rho$-dependence of (7.2) is a *local* estimate: measured first-stalling $\eta$ on a $0.01$
grid is $0.23$ / $0.21$ / $0.19$ / $0.19$ for $\rho=0.5$ / $1$ / $2$ / $4$, against predicted
$0.224$ / $0.202$ / $0.168$ / $0.126$ — accurate for $\rho\lesssim1$ and increasingly conservative
beyond, where the transient rather than the fixed point decides.

## 9. Stochastic gradient noise cannot rescue the run

Sections 1–8 use exact gradients. Real training uses a stochastic estimate; the question is
whether *unbounded* noise variance can kick the iterate out of $D$. It cannot — only the required
head start into $D$ grows with the noise level.

### 9.1 Setup

$D=\{s<0\}$ depends only on $s=w+b$, and by (2.1) $\partial\hat y/\partial w=\partial\hat y/\partial b$
identically (both equal $v\,\mathrm{relu}'(s)$) — so track $s$ directly rather than the full
$\theta=(w,b,v)$. Write $\partial_s\mathcal L(\theta,\lambda):=\partial_w\mathcal L(\theta,\lambda)+\partial_b\mathcal L(\theta,\lambda)$
for the derivative along $w+b$. Since $s^{k+1}=w^{k+1}+b^{k+1}=s^k-\alpha^k(\bar y_w^k+\bar y_b^k)$,
let $\bar y_s^k:=\bar y_w^k+\bar y_b^k$ be the resulting estimate of $\partial_s\mathcal L(\theta^k,\lambda^k)$
used at step $k$, and $\mathcal F^k=\sigma(s^1,\dots,s^k)$. Assume

$$E[\bar y_s^k\mid\mathcal F^k]=\partial_s\mathcal L(\theta^k,\lambda^k),\qquad
\operatorname{Var}(\bar y_s^k\mid\mathcal F^k)\le\sigma_s^2\ \text{ a.s.}, \tag{9.1}$$

together with Robbins–Monro step sizes

$$\alpha^k\ge0,\qquad\sum_{k=1}^\infty\alpha^k=\infty,\qquad\sum_{k=1}^\infty(\alpha^k)^2=:A_2<\infty. \tag{9.2}$$

$\sigma_s^2$ is otherwise arbitrary — it is not required to be small relative to the true gradient,
only finite.

### 9.2 A quantitative trap: Doob's inequality

By (3.1), $\partial_s\mathcal L\equiv0$ on $D$ for *every* $\lambda,\rho$ (it's the sum of the $w$- and
$b$-components of $\nabla_\theta\mathcal L$, both $\equiv0$ there), so as long as $s^1,\dots,s^k<0$,

$$\bar y_s^i=\zeta_s^i\ (i\le k),\qquad s^k=s^0-\underbrace{\textstyle\sum_{i=1}^k\alpha^i\zeta_s^i}_{=:M_k}, \tag{9.3}$$

where $\zeta_s^k:=\bar y_s^k-\partial_s\mathcal L(\theta^k,\lambda^k)$. $\{\zeta_s^k\}$ is a
martingale-difference sequence w.r.t. $\{\mathcal F^k\}$ — $E[\zeta_s^k\mid\mathcal F^k]=0$ and
$E[(\zeta_s^k)^2\mid\mathcal F^k]=\operatorname{Var}(\bar y_s^k\mid\mathcal F^k)\le\sigma_s^2$ a.s.,
both by the tower property applied to (9.1) — the standard remark used to set up SGD-noise
analyses. Since $\{\alpha^k\}$ is square-summable, this is exactly the setting of Condition A.4 in
[DacDruKakLee2018] (their Lemma 4.1): $M_k$ is a square-integrable $\{\mathcal F^{k+1}\}$-martingale
with predictable compensator

$$\langle M\rangle_j:=\sum_{k=1}^j E\big[(\alpha^k\zeta_s^k)^2\mid\mathcal F^k\big]\ \le\ \sigma_s^2\sum_{k=1}^j(\alpha^k)^2\ \le\ \sigma_s^2A_2\qquad\text{a.s.}$$

(the two inequalities hold termwise, pathwise, from (9.1)). Since $M_0=0$, telescoping
$M_k^2=\sum_{j\le k}[2M_{j-1}\Delta M_j+(\Delta M_j)^2]$ ($\Delta M_j:=M_j-M_{j-1}=\alpha^j\zeta_s^j$)
and noting the cross terms vanish in expectation — $M_{j-1}$ is $\mathcal F^j$-measurable and
$E[\Delta M_j\mid\mathcal F^j]=0$, so $E[2M_{j-1}\Delta M_j]=0$ by the tower property — gives
$E[M_k^2]=E[\langle M\rangle_k]$: an equality of *expectations* (numbers), since $\langle M\rangle_k$
is itself generally random. Taking $E[\cdot]$ of the a.s. bound above,

$$E[M_k^2]=E[\langle M\rangle_k]\ \le\ \sigma_s^2A_2\qquad\text{for every }k, \tag{9.4}$$

using (9.2)'s square-summability; the divergence $\sum\alpha^k=\infty$ plays no role here — it
matters only for making a *nonzero* drift accumulate, which is not the regime we're in. And since
$\langle M\rangle_\infty<\infty$ a.s., the martingale convergence theorem
([dembo2019lecture], Thm 5.3.33) gives

$$\lim_{j\to\infty}\sum_{k=1}^j\alpha^k\zeta_s^k=q\quad\text{a.s., a finite limit,} \tag{9.4'}$$

the classical Robbins–Monro summability lemma — normally the last step in proving a
stochastic-approximation method *converges*; here the drift is zero, so it says the run, while
trapped, actually settles at $s^0-q$ rather than merely staying bounded. The escape bound below
only needs (9.4), not the limit $q$.

Let $\tau=\inf\{k:s^k\ge0\}$ and $D_0=-s^0=\operatorname{dist}(\theta^0,\partial D)$. Before $\tau$,
exiting requires $M_k\ge D_0$ for some $k$. Doob's (Kolmogorov's) maximal inequality, applied to
$M_1,\dots,M_K$ for each finite $K$ and then letting $K\to\infty$ (the events
$\{\max_{k\le K}M_k\ge D_0\}$ increase to $\{\sup_kM_k\ge D_0\}$), gives

$$P(\tau<\infty)\ \le\ P\Big(\sup_k M_k\ge D_0\Big)\ \le\ \frac{\sigma_s^2A_2}{D_0^2}. \tag{9.5}$$

$$\boxed{\ P(s^k<0\ \forall k\ge0)\ \ge\ 1-\frac{\sigma_s^2A_2}{D_0^2}\ } \tag{9.6}$$

For *every* noise level $\sigma_s^2<\infty$, however large, and every confidence $1-\epsilon$,
starting at $D_0>\sigma_s\sqrt{A_2/\epsilon}$ traps the run forever with probability $\ge1-\epsilon$.
Larger noise only requires starting further in — proportionally, since $D_0$ scales linearly in
$\sigma_s$ — it never defeats the trap.

### 9.3 The dual variable's own noise changes nothing

$\lambda^k$ follows its own noisy update, e.g. $\lambda^{k+1}=[\lambda^k+\gamma^k(g(\theta^k)+\eta^k)]_+$,
and enters the primal only through the coefficient $c(\lambda^k,\rho,g(\theta^k))$ multiplying
$\partial_sg\equiv0$ (2.1). Whether $\lambda^k$ diverges (3.2), oscillates, or is itself driven by
unbounded-variance noise $\eta^k$ is irrelevant: $c\cdot\partial_sg\equiv0$ on $D$ regardless, so
(9.3)–(9.6) hold unchanged with a noisy dual loop running alongside the noisy primal one.

### 9.4 Verified

Simulated the full $(w,b,v)$ HPR-ALM dynamics directly (not the scalar recursion in $s$ alone):
i.i.d. Gaussian noise of per-coordinate standard deviation $\sigma$ added to each of $\bar y_w^k,
\bar y_b^k,\bar y_v^k$ *and* independently to the constraint value feeding the dual update. Since
$\bar y_s^k=\bar y_w^k+\bar y_b^k$ is the sum of two independent noise draws of variance $\sigma^2$,
this instantiates (9.1) with $\sigma_s=\sqrt2\,\sigma$. Step size $\alpha^k=0.3/k$
($A_2=0.09\,\pi^2/6=0.148044$), $\rho=1$, $\gamma=0.1$, $2000$–$4000$ seeds per cell, run to
$k=20{,}000$ (increments below double precision by then, so this is indistinguishable from
$k=\infty$):

| $\sigma$ | $\sigma_s=\sqrt2\,\sigma$ | $D_0$ | bound (9.6) | measured escape frac. |
|---|---|---|---|---|
| $1$  | $1.41$  | $2$  | $\le0.0740$ | $0.0000$ |
| $1$  | $1.41$  | $20$ | $\le0.0007$ | $0.0000$ |
| $3$  | $4.24$  | $2$  | $\le0.6662$ | $0.1780$ |
| $3$  | $4.24$  | $5$  | $\le0.1066$ | $0.0025$ |
| $3$  | $4.24$  | $20$ | $\le0.0067$ | $0.0000$ |
| $10$ | $14.14$ | $2$  | $\le1.0000$ | $0.5075$ |
| $10$ | $14.14$ | $5$  | $\le1.0000$ | $0.2820$ |
| $10$ | $14.14$ | $10$ | $\le0.2961$ | $0.0625$ |
| $10$ | $14.14$ | $20$ | $\le0.0740$ | $0.0000$ |

$\sigma=10$ ($\sigma_s=14.14$) is roughly $4$–$5\times$ the true gradient norm at $\theta_0$ (Sec. 6,
$\|\nabla f(\theta_0)\|\approx2.5$–$3.0$) — noise dominating signal by a wide margin. Target: $99\%$
confidence ($\epsilon=0.01$) at $\sigma_s=14.14$ needs, by (9.6), $D_0\ge14.14\sqrt{0.148044/0.01}=54.41$;
measured escape fraction there, $0$ out of $4000$ seeds — the bound holds (and is conservative, as
Doob's inequality generally is).
