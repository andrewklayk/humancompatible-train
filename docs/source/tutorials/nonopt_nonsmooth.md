---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: hc-dev
  language: python
  name: python3
---

# Nonsmooth Training with `NonOpt`

`NonOpt` (from `humancompatible.train.sqp`) is a PyTorch port of
[NonOpt](https://frankecurtis.github.io/NonOpt/) (Curtis & Zebiane,
[arXiv:2503.22826](https://doi.org/10.48550/arXiv.2503.22826)) — a solver for
**unconstrained** minimization of locally Lipschitz, possibly nonconvex and **nonsmooth**
objectives.

It is the odd one out in this package. The other methods here (`ALM`, `PBM`, `GhostSQP`, …)
exist to handle *constraints*; `NonOpt` handles a *kinked objective*. Reach for it when the
thing you minimize is not differentiable at the point you want to end up at — worst-case
losses, hinge/$\ell_1$ terms, minimax formulations — and when the problem is small enough to
train **full-batch and deterministically**.

At every iteration it

1. collects (sub)gradients from nearby previously visited points into a **bundle**,
2. solves a small convex QP over the convex hull of that bundle to get a search direction
   (a proximal-bundle / cutting-plane step) preconditioned by a **self-correcting BFGS**
   inverse-Hessian approximation,
3. takes a **weak Wolfe** line search step, and
4. shrinks a *stationarity radius* until approximate stationarity can be certified.

The interface mirrors `torch.optim.LBFGS`: you pass a closure, and a single `step()` may call
it several times (bundle points and line-search trials). Unlike the rest of the package, it
does **not** work with minibatches — it needs the same deterministic function every call.

The task below is **minimax-fair (group-DRO) classification**: minimize the *worst* group's
loss rather than the average loss. That objective is a pointwise maximum, so it is nonsmooth
exactly where its minimizer lives.

+++

## Load the data

Same ACS Income setup as the [Basic Usage](basic_usage.md) tutorial (and the GhostSQP
fairness tutorial): predict whether income exceeds \$50k, with
gender as the sensitive attribute. Feel free to skip this cell.

```{code-cell} ipython3
import numpy as np
import torch
from folktables import ACSDataSource, ACSIncome, generate_categories
from sklearn.preprocessing import StandardScaler

# NonOpt is a deterministic solver and its line search tests small decreases;
# double precision keeps those tests meaningful.
torch.set_default_dtype(torch.float64)

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["FL"], download=True)
definition_df = data_source.get_definitions(download=True)
categories = generate_categories(features=ACSIncome.features, definition_df=definition_df)
df_feat, df_labels, _ = ACSIncome.df_to_pandas(acs_data, categories=categories, dummies=True)

sens_cols = ["SEX_Female", "SEX_Male"]
features = df_feat.drop(columns=sens_cols).to_numpy(dtype=np.float64)
features = StandardScaler().fit_transform(features)
labels = df_labels.to_numpy(dtype=np.float64).ravel()
female = df_feat["SEX_Female"].to_numpy(dtype=np.float64)

# Subsample: NonOpt is full-batch, so every closure call touches all of these rows.
sel = np.random.default_rng(0).choice(len(features), size=5_000, replace=False)
X = torch.tensor(features[sel])
y = torch.tensor(labels[sel]).unsqueeze(1)

# Four groups: gender x true label.  Worst-group loss over these cells is the usual
# group-DRO / minimax-fairness objective.
group_id = (2 * torch.tensor(female[sel]) + y.squeeze(1)).long()
GROUPS = ["male / <=50k", "male / >50k", "female / <=50k", "female / >50k"]
masks = [group_id == k for k in range(len(GROUPS))]

print(f"{X.shape[0]} samples, {X.shape[1]} features")
for name, mask in zip(GROUPS, masks):
    print(f"  {name:<16s} n = {int(mask.sum())}")
```

## The nonsmooth objective

With $\mathcal{G}$ the four groups and $L_g(w)$ the cross-entropy loss on group $g$, we
minimize

$$ f(w) \;=\; \max_{g \in \mathcal{G}} L_g(w). $$

Each $L_g$ is smooth, but their pointwise maximum is not: $f$ has a kink wherever two groups
tie for worst. And that is precisely where the minimizer sits — balancing the groups is the
whole point of the formulation — so the nonsmoothness is *active at the solution*, not an
incidental detail. A subgradient step there is the gradient of whichever single group happens
to be worst at that iterate, which is why plain first-order methods chatter between groups
instead of settling.

We use a linear model, so each $L_g$ is convex and so is $f$.

```{code-cell} ipython3
bce = torch.nn.BCEWithLogitsLoss()


def make_model():
    """Linear classifier, zero-initialized (so every optimizer starts identically)."""
    model = torch.nn.Linear(X.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    return model


def group_losses(model):
    """Per-group cross-entropy losses, shape (4,)."""
    logit = model(X)
    return torch.stack([bce(logit[mask], y[mask]) for mask in masks])


def worst_group_loss(model):
    return group_losses(model).max()
```

## Driving `NonOpt`

Exactly the `torch.optim.LBFGS` protocol: a closure that zeroes the gradients, evaluates the
objective, calls `backward()` and returns the loss. Autograd supplies *a* subgradient of the
`max` (the gradient of the currently-worst group), which is all a bundle method needs.

Two things to note:

* `step()` calls the closure **several times** — for the tentative gradient step, the bundle's
  trial points, and the line search. So we count *gradient evaluations*, not steps, to compare
  fairly against `Adam` (one per step) and `LBFGS`.
* `NonOpt` decides when it is done. `optimizer.converged` becomes `True` once the solver
  certifies approximate stationarity (`status == "stationary"`) or stops making progress at
  the final stationarity radius (`status == "objective_similarity"`).

```{code-cell} ipython3
def train(make_optimizer, max_steps, verbose=False):
    """Runs an optimizer on the worst-group loss; returns (model, optimizer, history)."""
    model = make_model()
    optimizer = make_optimizer(model.parameters())
    n_evals = 0

    def closure():
        nonlocal n_evals
        n_evals += 1
        optimizer.zero_grad()
        loss = worst_group_loss(model)
        loss.backward()
        return loss

    with torch.no_grad():
        history = [(0, float(worst_group_loss(model)))]

    for step in range(1, max_steps + 1):
        optimizer.step(closure)
        with torch.no_grad():
            history.append((n_evals, float(worst_group_loss(model))))
        if verbose and step % 25 == 0:
            print(f"  step {step:4d}  evals {n_evals:5d}  worst-group loss {history[-1][1]:.6f}")
        if getattr(optimizer, "converged", False):  # NonOpt only
            if verbose:
                print(f"  stopped after {step} steps: status = {optimizer.status}")
            break

    return model, optimizer, history
```

```{code-cell} ipython3
from humancompatible.train.sqp import NonOpt

model_nonopt, solver, hist_nonopt = train(lambda p: NonOpt(p), max_steps=600, verbose=True)

print(f"\nconverged            {solver.converged}  ({solver.status})")
print(f"stationarity radius  {solver.stationarity_radius:.1e}")
print(f"worst-group loss     {hist_nonopt[-1][1]:.6f} after {hist_nonopt[-1][0]} gradient evaluations")
```

The solver stops on its own, and the per-group losses come out **equalized** — the signature of
a minimax solution, and the reason the objective is nonsmooth there:

```{code-cell} ipython3
with torch.no_grad():
    for name, value in zip(GROUPS, group_losses(model_nonopt).tolist()):
        print(f"  {name:<16s} loss = {value:.6f}")
```

## Baselines: `Adam` and `LBFGS`

Both are given a *larger* budget of gradient evaluations than `NonOpt` used.

```{code-cell} ipython3
model_adam, _, hist_adam = train(lambda p: torch.optim.Adam(p, lr=1e-2), max_steps=3000)
model_lbfgs, _, hist_lbfgs = train(
    lambda p: torch.optim.LBFGS(p, line_search_fn="strong_wolfe"), max_steps=300
)

RUNS = [
    ("NonOpt", model_nonopt, hist_nonopt),
    ("Adam", model_adam, hist_adam),
    ("LBFGS", model_lbfgs, hist_lbfgs),
]

print(f"{'method':<8} {'final f':>10} {'best f':>10} {'grad evals':>11}")
for name, _, hist in RUNS:
    print(f"{name:<8} {hist[-1][1]:>10.6f} {min(v for _, v in hist):>10.6f} {hist[-1][0]:>11d}")
```

`Adam`'s **final** iterate is much worse than the best one it visited: subgradient steps at a
kink overshoot, and the iterate oscillates between whichever group is momentarily worst. No
stopping test would have told you to keep the good one. `LBFGS` does reach a comparable value
on this convex problem, but never terminates by itself — it simply exhausts the step budget.

Counting the gradient evaluations each method needs to first reach a given worst-group loss:

```{code-cell} ipython3
print(f"{'target f':>10} " + " ".join(f"{name:>9}" for name, _, _ in RUNS))
for target in (0.3600, 0.3560, 0.3550, 0.3545, 0.3543):
    cells = []
    for _, _, hist in RUNS:
        hit = next((evals for evals, value in hist if value <= target), None)
        cells.append(f"{hit:>9d}" if hit is not None else f"{'never':>9}")
    print(f"{target:>10.4f} " + " ".join(cells))
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Categorical slots 1-3 of a validated palette (blue / orange / aqua).
COLORS = {"NonOpt": "#2a78d6", "Adam": "#eb6834", "LBFGS": "#1baf7a"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

LABEL_OFFSETS = {"NonOpt": (4, 9), "Adam": (4, 0), "LBFGS": (4, 0)}
for name, _, hist in RUNS:
    evals, values = zip(*hist)
    ax1.plot(evals, values, color=COLORS[name], lw=2, label=name)
    ax1.annotate(name, (evals[-1], values[-1]), color=COLORS[name],
                 xytext=LABEL_OFFSETS[name], textcoords="offset points", va="center", fontsize=9)

ax1.set_xlim(0, 4300)
ax1.set_ylim(0.3535, 0.3865)
ax1.set_xlabel("gradient evaluations")
ax1.set_ylabel(r"worst-group loss  $\max_g L_g$")
ax1.set_title("Convergence (endgame; every run starts from $f = 0.693$)")
ax1.legend(frameon=False, loc="upper right")
ax1.grid(axis="y", alpha=0.25)
ax1.set_axisbelow(True)
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

# Dot plot rather than bars: the interesting differences are far from zero, and bars must
# keep a zero baseline.
positions = np.arange(len(GROUPS))[::-1]
per_group = {name: group_losses(model).detach().tolist() for name, model, _ in RUNS}
for row, position in enumerate(positions):
    spread = [values[row] for values in per_group.values()]
    ax2.plot([min(spread), max(spread)], [position] * 2, color="#cccccc", lw=1, zorder=1)
# NonOpt and LBFGS land on nearly the same numbers, so LBFGS gets an open ring drawn around
# the filled dots instead of hiding them.
MARKS = {"NonOpt": dict(marker="o", ms=8, markeredgecolor="white", markeredgewidth=1.5),
         "Adam": dict(marker="s", ms=7.5, markeredgecolor="white", markeredgewidth=1.5),
         "LBFGS": dict(marker="o", ms=14, markerfacecolor="none", markeredgewidth=2)}
for name, values in per_group.items():
    style = dict(MARKS[name])
    ax2.plot(values, positions, color=COLORS[name], label=name, ls="none", zorder=2,
             markeredgecolor=style.pop("markeredgecolor", COLORS[name]), **style)

ax2.set_yticks(positions)
ax2.set_yticklabels(GROUPS)
ax2.set_ylim(-0.6, len(GROUPS) - 0.1)
ax2.set_xlabel("group loss at the final iterate")
ax2.set_title("Per-group losses")
ax2.legend(frameon=False, ncols=3, loc="lower left", handletextpad=0.4, columnspacing=1.2)
ax2.grid(axis="x", alpha=0.25)
ax2.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax2.spines[side].set_visible(False)
ax2.tick_params(axis="y", length=0)

fig.tight_layout()
plt.show()
```

## Choosing the strategies

The three components can be swapped independently. `cutting_plane` (proximal bundle) is the
reference default; `gradient_combination` samples extra subgradients within the stationarity
radius (gradient sampling); `gradient` drops the bundle entirely and is just a self-correcting
quasi-Newton step. Line searches are `weak_wolfe` (default) and `backtracking`; the inverse
Hessian is `limited_memory` (default) or `dense` (only for small models).

```{code-cell} ipython3
import time

VARIANTS = [
    ("cutting_plane + weak_wolfe (defaults)", {}),
    ("gradient_combination", {"direction": "gradient_combination",
                              "direction_options": {"random_sample_factor": 3}}),
    ("gradient", {"direction": "gradient"}),
    ("cutting_plane + backtracking", {"line_search": "backtracking"}),
]

print(f"{'variant':<38} {'final f':>10} {'evals':>7} {'status':>22} {'time':>7}")
for label, options in VARIANTS:
    start = time.perf_counter()
    _, opt, hist = train(lambda p, o=options: NonOpt(p, **o), max_steps=600)
    print(f"{label:<38} {hist[-1][1]:>10.6f} {hist[-1][0]:>7d} "
          f"{opt.status:>22} {time.perf_counter() - start:>6.1f}s")
```

All variants land on the same solution here; they differ in cost per step. The bundle QP is
solved over the convex hull of the collected subgradients, so `cutting_plane` and
`gradient_combination` pay for extra closure calls and a larger QP, which buys robustness on
harder kinks than this one — on this problem the cheap tentative gradient step usually already
gives sufficient decrease, so the bundle rarely has to engage.

+++

## Takeaways

* **`NonOpt` optimizes a nonsmooth objective to a certified stationary point.** On the
  minimax-fair task it equalized all four group losses and stopped on its own with
  `status = "objective_similarity"` at stationarity radius $10^{-4}$.
* **Compare on gradient evaluations, not steps.** One `NonOpt` step calls the closure several
  times. Even so, it reached the tight tolerances in fewer evaluations than `LBFGS` (which is
  ahead of it in the middle of the run), and `Adam` never reached them at all.
* **`Adam` has no business here.** Its last iterate was substantially worse than its best, with
  no signal to distinguish them — the failure mode of subgradient descent at an active kink.
  `LBFGS` is competitive on this *convex* problem, but at a higher evaluation count and with no
  termination test.
* **Know the limits.** Full-batch and deterministic only (never hand it minibatches), a single
  parameter group, and each step costs a QP solve over the bundle — so it suits small-to-medium
  models. For *constrained* problems use the dual optimizers or `GhostSQP` instead; `NonOpt`
  solves unconstrained problems only.
