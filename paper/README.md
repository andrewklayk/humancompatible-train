# Paper experiments

Experiments for the MPC software paper on `humancompatible-train`. Deliberately
plain scripts rather than a framework: each one is meant to be read, edited and
debugged directly.

Two conventions run through all of them.

**Every experiment states its predictions in code, before its results.** Each
script's module docstring lists numbered predictions; `register_predictions`
evaluates them and `--check` makes the process exit non-zero if one fails. A
prediction that turns out to be wrong is reported as wrong rather than quietly
edited — where that happened, the docstring says so and both the original and the
corrected claim are kept. **E3 is the deliberate exception**: it is a measurement of
throughput and achieved sparsity on hardware, with no thresholds worth asserting in
advance, so it registers nothing and its `--check` exits 0.

**Artifacts are committed.** Figures are PDF (the repo's `.gitignore` excludes
`*.png`/`*.svg`, and LaTeX wants PDF anyway), tables are Markdown plus a JSON
sidecar, raw per-iteration data is CSV. `paper/results/**` is explicitly
re-included in `.gitignore` so all of it can go into the reproducibility artifact.

## Running

```bash
bash paper/e0/run_all.sh                 # everything, writes paper/results/e0*/
bash paper/e0/run_all.sh --quick         # seconds; artifacts to a temp dir
bash paper/e0/run_all.sh --check         # non-zero exit on a violated prediction
bash paper/e2/run_all.sh --check         # likewise for E2
bash paper/e3/run_all.sh --quick         # E3 on the CPU stub model, no downloads
python paper/e0/a_multipliers.py --help  # per-script options
```

`HC_PAPER_RESULTS=<dir>` redirects all artifacts, which is how `--quick` and
`tests/test_distributed.py` avoid overwriting a real run's results.

Requires `qpsolvers` (with `clarabel`), `scikit-learn`, `matplotlib` and `scipy`
on top of the package's own dependencies.

## E0 — validation

E0 is the faithfulness section: before any expensive experiment, show that each
shipped implementation does what its reference says. The algorithms are other
people's; the implementation is ours, so this is the part that earns trust in it.

| script | claim | runtime |
|---|---|---|
| [e0/a_multipliers.py](e0/a_multipliers.py) | the dual optimizers are faithful to the mathematics they implement | ~10 min (`--quick` skips convergence: seconds) |
| [e0/b_nonopt.py](e0/b_nonopt.py) | `NonOpt` reaches the published optimal values on the ten standard nonsmooth problems at n = 50 | ~10 min |
| [e0/d_distributed.py](e0/d_distributed.py) | data-parallel equivalence: duals identical across ranks, `G x B` = `1 x (G*B)` where that can hold, and the measured gap where it cannot | ~1 min |

### E0a — mathematical faithfulness

The question is whether each dual optimizer encodes the surrogate and multiplier
update its reference specifies. That is an **algebraic** property, so E0a tests it
algebraically. Convergence is a separate question — it depends on step sizes,
conditioning and tuning — and is reported in one untuned configuration per method
rather than as a ranking.

Four categories of claim, in decreasing strength.

**F — fixed-point consistency.** At a KKT point `(x*, y*)` a faithful method must be
stationary: one `forward_update` leaves the multipliers put and `∇ₓ` of the
surrogate vanishes. No step size, no tuning, no iteration. It is also **sharp** —
switching `ALM`'s quadratic term back to raw `c` on inequality data takes `∇ₓ` from
1.8e-15 to 4.7 on `qp_inactive` — so F is a live regression guard for the `[c]₊`
fix, asserted as such. That is the same defect the superseded design needed 20 000
tuned iterations to expose.

Where a method's declared `lower_bound` is positive, `y*ᵢ = 0` is unrepresentable.
The resulting deviation is **computed from the bound** rather than tolerated: PBM's
multiplier drift is exactly its 1e-4 floor, and the induced `∇ₓ` offset is allowed
at `‖J‖ · ‖y − y*‖`, first-order through the Jacobian.

**R — exact reductions between methods.** Three of the four classes reduce to `ALM`,
which tests them against an independent implementation instead of against a
tolerance:

| | reduction | preconditions | bar |
|---|---|---|---|
| R1 | `nuPI(kp=0, ki=γ)` ≡ `ALM(lr=γ, ρ=0)` | none | bitwise |
| R2 | `iALM(β, σ=1, γ→∞)` ≡ `ALM(lr=β, ρ=β)` | `γ ≥ β‖c‖`, so the safeguard doesn't bind | bitwise |
| R3 | `PBM(penalty_update="alm", ρₚ, γ)` ≡ `ALM((1−γ)/ρₚ, 1/ρₚ)` | `p₀ = y₀ρₚ`; `c/p ≥ −0.5` | rounding |

R3 is the strongest single assertion here: with `p = yρₚ` the quadratic-logarithmic
barrier `Σ yᵢpᵢφ(cᵢ/pᵢ)` collapses to `y'c + ‖c‖²/2ρₚ`, so one comparison validates
PBM's barrier algebra, its penalty update and its dual rule together.

The `bar` column matters. R1 and R2 execute *the same* floating-point operation
(`duals.add_(c, alpha=·)`), so bitwise equality is the right test. R3 computes
`y(γ + (1−γ)(1 + c/p))` against `y + lr·c` — the same number by a different route,
which differs in the last bit — so it is held to rounding. Each reduction is checked
for one step and along a trajectory, and the trajectory **stops at the first iterate
where a precondition breaks**, reporting that step. Those boundaries are results in
themselves: R2 survives 200 steps on the convex problems but only 117 on
`qp_nonconvex` before `iALM`'s `γ` safeguard binds; R3 survives 200 on `qp_active`
but 6 on `qp_inactive` and 3 on `svm_iris` before leaving `quad_log`'s quadratic
branch.

**I — invariances.** I1: scaling the constraints by α must divide the multipliers by
α and change nothing else. I2: an equality `h = 0` posed as `h ≤ 0, −h ≤ 0` must
recover `x*` and `y⁺ − y⁻ = y_eq` — a reduction the package's problem statement
relies on and nothing else tested. Individual `y⁺`/`y⁻` are not determined, so only
the difference is asserted, and `qp_equality`'s reference multipliers are given
deliberately alternating signs, since a drawn multiplier of magnitude 1e-3 would
test nothing.

`PBM` is **exempt from I2**, structurally and not incidentally: the two-sided form
has no strictly feasible interior, because at any `x` one of each pair is violated.
PBM's multiplicative update then grows whichever side is violated and alternates —
measured ping-ponging between its 1e-4 floor and its 100 ceiling. Its documented
route to an equality is the threshold `|h| ≤ τ`, which keeps an interior but poses a
*different* problem with different multipliers, so it is not comparable and is not
attempted. The observed divergence is reported, not gated.

**C — convergence, one untuned configuration per method.** Primal step
`1/(L_f + ρ‖J‖²)`, dual step `1/‖J‖²`, no sweep. The iteration budget is *derived*
rather than fixed: the number of dual updates needed to carry `y` from 0 to `y*`
scales like `‖J‖²`, so a count that converges on `qp_active` (`‖J‖² = 19`) is 16×
too small for `svm_iris` (`‖J‖² = 305`). Every problem gets `700/dual_step`
iterations, which is equal *progress* rather than equal iterations. Measured on
`svm_iris` at its derived step: 1.6e-03 relative KKT after 20k, 4.7e-06 after 50k,
2.8e-10 after 100k, 4.6e-14 after 200k.

Expectations are stated per problem and conditioned on a **structural predicate**,
never on a method name:

| problem | property | expectation |
|---|---|---|
| `qp_active` | strongly convex, all active, `y*>0`, LICQ, strict complementarity ⟹ unique KKT point | **every** method converges; a failure is a defect |
| `qp_inactive` | adds exact zeros in `y*` | every method with `lower_bound == 0` reaches `y*ᵢ = 0` exactly; one with `lower_bound > 0` cannot, and its error is bounded below by that bound |
| `svm_iris` | objective Hessian **singular** in the bias ⟹ Lagrangian bilinear in `(b, y)` | plain primal-dual ascent has no last-iterate guarantee; methods supplying curvature (`ρ>0`) or damping (`kp>0`) converge, one with neither is not expected to |
| `qp_nonconvex` | no convergence theory for a fixed-penalty Lagrangian on an indefinite objective | **nothing is claimed** |

**Most configurations fail on the nonconvex problem, and that is the main result of
that panel** rather than an incidental detail. A fixed-penalty Lagrangian surrogate
is unbounded below on an indefinite objective and none of these methods' assumptions
rule that out, so failure is a limit of applicability. It is not an artifact of the
derived step size either — `max |eig(Q)|` is a valid gradient-Lipschitz constant for
an indefinite quadratic, so the step is legitimate and the iterates leave anyway.
`results/e0a/e0a_status.md` gives the counts, and the `status` column is three-way
(`solved` / `bounded` / `diverged`) because a two-way split reported an iterate at
violation 36 as "ok".

**Reference values are exact, not solver output.** The QPs are constructed backwards
from a chosen `(x*, y*)` and cross-checked against clarabel; `svm_iris` uses clarabel
only to identify the active set and then solves the resulting KKT system exactly.
Without that, the solver's own ~1e-9 error becomes a floor that several methods tie
at for reasons that say nothing about them — which is precisely what happened in the
superseded design.

**Tolerances: one rule per category**, replacing the four unrelated constants the
superseded design accumulated. F, R and I are algebraic identities, so they assert
bitwise where the operations are literally identical and `64·eps·scale` otherwise —
anything above rounding is a bug. C reports a *relative* KKT residual against a
single tolerance.

#### Superseded predictions

E0a previously tested convergence as a proxy for faithfulness, under predictions
P1–P8. The reasoning errors are worth keeping even though they are no longer live
gates:

- **P1** (`ALM(ρ=1)` and `nuPI(ρ=0)` reach 1e-4) was mis-scoped for `nuPI`: the
  property that carries it is *having a quadratic term*, which `nuPI` does not, so it
  inherits rather than escapes the bilinear-bias obstruction. Now covered by F plus
  C's `svm_iris` row.
- **P4** (`iALM`'s `β` coupling is harmful) was withdrawn: at `β=0.1` it matches
  `ALM(ρ=1)` at machine precision. The contrary reading was an artifact of driving
  the loop with `forward` + `update` instead of `forward_update`.
- **P5a/P5b** existed only because the thresholds were unprincipled — 1e-4 on
  `‖y−y*‖` against 1e-8 on the violation demands feasibility be 10⁴ times tighter.
  The relative KKT residual in C replaces both.
- **P6** (a quadratic coefficient above `−λ_min(Q)` is what keeps `qp_nonconvex`
  bounded) was wrong twice: "bounded" was the wrong property, and the argument only
  bounds the surrogate at *fixed* `y`, whereas rising multipliers can hold the
  iterates in the box. `nuPI(ρ=0)` settled it — no quadratic term at all, and it
  stays bounded.
- **P7** survives as R1, alongside two stronger siblings.

### E0b — NonOpt against published values

The ten Karmitsa problems at n = 50. Formulae come from Karmitsa's report and the
targets from the published table, so **transcription is verified rather than
assumed**: `paper/problems/nonsmooth.py::verify` checks the implemented `f(x0)`
against a hand-derived closed form to 1e-10 *and* the closed form against the
published value to the table's printed precision. A mistranscribed formula fails
at the start of the run instead of looking like a solver failure.

Two of our defaults differ from the reference configuration and are swept rather
than hidden: `inverse_hessian` (`dense` matches the reference's full-memory BFGS;
our default is `limited_memory`) and `point_set_options["size_factor"]` (the
shipped 0.05 keeps only **two** bundle points at n = 50, and zero below n = 20).

Three interface facts are worked around and reported, since users will hit them:
there are no evaluation counters (the closure is wrapped, and one call yields both
value and gradient, so `#func` and `#grad` cannot be separated); `step()` returns
the loss only on the first call; and the stationarity test scales with the initial
gradient, so on `maxq` the effective tolerance is 1e-2 rather than the requested
1e-4. Both tolerances are in the results table.

The direct comparison against the C++ NonOpt is deferred. The reference solvers'
own final values (Curtis & Que Tables 2-4) are **not yet transcribed** — see
`REFERENCE_SOLVERS_N50` in [problems/nonsmooth.py](problems/nonsmooth.py); fill it
in and the summary table grows those columns automatically. Until then the
comparison is against the published `f*` only.

### E0d — data-parallel equivalence

Two gloo ranks against one process at the pooled batch size. The headline is that
equivalence is **conditional**, and the conditions are worth stating in the paper:

- Duals are bitwise identical on every rank, always.
- A surrogate **linear in the constraint vector** (`ALM(rho=0)`, `nuPI(rho=0)`)
  reproduces the pooled-batch run exactly, because `ReduceOp.AVG` of per-rank
  means is the mean over the union and DDP averages gradients.
- A surrogate with a **quadratic term** does not, since
  `mean_r ||[c_r]_+||^2 != ||[mean_r c_r]_+||^2`. After one step only the
  parameters differ — the dual update sees only the reduced vector — and over more
  steps that discrepancy propagates into the duals through the constraint values.
- A **ratio-type** constraint (the per-group rates E2 and E3 use) is a nonlinear
  functional of expectations, so nothing matches exactly regardless of the
  surrogate. The gap is measured against per-rank batch size, which is what bounds
  how small a per-rank batch those experiments may use.

## E2 — fairness-constrained learning on real data

E2 is where the constraint is a statistic of *data* rather than of parameters, which
buys two things no synthetic problem can give: the constraint's **generalization**
behaviour, and a setting in which the cross-rank reduction actually has work to do.

| script | claim | runtime |
|---|---|---|
| [e2/a_fairness.py](e2/a_fairness.py) | the methods deliver the constraint on real data, and the violation they deliver on *train* is not the violation you get on *test* | ~35 min (`--quick`: ~1 min) |
| [e2/b_parallel.py](e2/b_parallel.py) | `G x B` = `1 x (G*B)` exactly — but only under balanced sharding, and only for a surrogate linear in `c` | ~2 min |

Datasets: **ACSIncome** (folktables, 2018 1-Year) with the sensitive attribute the
*cross product* of marital status and sex (6 groups), and the **Dutch census 2001**
with sex x age (18 groups). Two constraint shapes, both from
[new_bench/constraints.py](../benchmark/new_bench/constraints.py) rather than
reimplemented: `pairwise`, the positive-rate gap over every ordered pair of groups
(`m = G(G-1)`, so 30 and 306), and `agg`, one aggregated fairret norm-loss (`m = 1`).

**Data is loaded offline.** Only the ACS states whose `psam_p*.csv` is already under
`benchmark/new_bench/data/` can be used — currently **FL and VA**, not the five the
plan once assumed; `fairness.available_states()` reports what is present. The Dutch
loader is pointed at the parquet in `benchmark/cache/datasets/`, because
`fairml_datasets` hardcodes a *cwd-relative* `Path("cache")` with no environment
override and its `dataset` module binds that path by from-import — so without the
redirect, running from anywhere but `benchmark/` silently re-downloads the ARFF.

**`SEX x RAC1P` is available but not the default.** Crossed with sex, RAC1P's tail
groups have as few as 5 rows in one state, and `BalancedBatchSampler` would then put
one row of such a group in every batch — a per-group rate estimated from one sample
is noise, not a constraint. `min_group` drops groups below a threshold and says so in
the problem's `notes`.

**Batch size is derived, not fixed.** `BalancedBatchSampler` puts
`batch_size // n_groups` of each group in every batch, so a batch size that gives 8
samples per group at 6 groups gives 2 at 18. E2 fixes the *per-group* count
(`PER_GROUP = 8`) and derives the batch size from it.

### E2a — do the methods deliver the constraint, and does it generalize?

Configurations are deliberately **untuned and identical across methods** — one primal
step, one dual step, everywhere — following E0a's convergence panel: the comparison is
then about the update rule rather than about who was tuned harder. This is not a
leaderboard and the script does not claim one. Tuning burden needs its own sweep and is
not answered here.

Metrics are evaluated **full-batch at the frozen end-of-epoch iterate**, never as an
epoch-mean of minibatch values, because the minibatch gradient norm has a noise floor
that never reaches zero — the methodological point inherited from `new_bench`.

Two design choices worth stating because they were arrived at by getting them wrong
first:

- **The timing control.** Plain Adam never evaluates the constraint, so timing dual
  methods against it measures the *fairret statistic* — a property of the problem —
  and not the dual layer at all. `Adam (c in graph)` pays the identical constraint
  forward and backward through a zero-weighted term, so its gradient is
  mathematically Adam's and the gap to a dual method isolates the multiplier update.
  That it lands on the same iterate as plain Adam is asserted, not assumed.
- **Wall-clock is reported in absolute time per step, not as a ratio.** At this model
  size a ratio is dominated by per-step interpreter overhead, and thresholding it
  would be arbitrary. The dual update is a fixed handful of small tensor ops, so the
  bound comes from that operation count. The honest reading: on an 808-64-32-1 MLP a
  few hundred microseconds *is* a visible fraction of a step, so the dual layer is not
  invisible here — the claim is that the cost is fixed in the model size, which is why
  E3 measures the same thing at 0.5 B parameters.

### E2b — data-parallel correctness, on a constraint the reduction changes

This is the claim **E3 cannot make**. E3's sparsity constraint is a closed form in the
gate parameters, so every rank computes the same value and `all_reduce(c, AVG)`
averages identical numbers — it cannot be caught being wrong. Here the per-rank values
genuinely differ and the reduction has to turn G per-shard estimates into the pooled
one.

The pivot is a fact about `fairret`'s `PositiveRate` that is easy to miss. It is a
`LinearFractionalStatistic` with `denom_slope = 0`, so per group it is a *ratio of
sums* — which E0d found never reproduces a pooled run exactly. But under **balanced**
batching the denominator is exactly `PER_GROUP` on every rank, a constant, so the
statistic collapses to a plain mean and becomes linear in the sample. **The
`BalancedBatchSampler` that per-group constraints already require for statistical
reasons is the same thing that makes data-parallel equivalence exact.** Those two facts
were previously unconnected, and the negative control (random sharding, where the
per-group counts differ and exactness is lost) is what shows the reduction is
load-bearing rather than decorative.

Batches are built explicitly rather than drawn from a sampler, because everything here
rests on one invariant — the union of the ranks' batches is the single-process batch —
and constructing it by hand makes that obvious instead of a property to be trusted
about a shuffler. It also means E2b does not need the distributed-aware sampler the
library still lacks; what it shows is *why* that sampler has to shard per group.

## E3 — sparsity-constrained LM fine-tuning, at scale

E3 is the one experiment run at a size that needs several GPUs, and that is why it exists:
E2's model is an 808-64-32-1 MLP, where a few hundred microseconds of multiplier update is
a visible share of a step. The question here is what the same layer costs when the model
is 0.5 B parameters and the step is real work.

| script | question | runtime |
|---|---|---|
| [e3/sweep.py](e3/sweep.py) | does each method land on a requested density, what is the model worth there, and what did it cost | hours (`--quick`: ~1 min) |
| [e3/scaling.py](e3/scaling.py) | how throughput and the dual layer's share move over 1 → 2 → 4 GPUs | ~20 min (`--quick`: ~1 min) |

The formulation is Eq. (3) of Gallego-Posada, Ramirez, Erraqabi, Bengio &
Lacoste-Julien, *Controlled Sparsity via Constrained Optimization* (NeurIPS 2022,
arXiv:2208.04425): structured hard-concrete L0 gates, one per MLP intermediate channel and
one per query head, with the expected parameter density constrained per layer. Their Eq. (5)
projected gradient descent-ascent **is** `ALM(penalty=0, is_ineq=True)` and their Eq. (6)
dual restart **is** `restart=True`, so the package already ships their method and it appears
in the sweep as `alm_gda` / `alm_gda_restart`. What is extended: their experiments are
vision (CIFAR, TinyImageNet, ImageNet), they assert *"negligible computational overhead"*
without measuring it, and their "Choice of optimizers" names the multiplier update as open
future work — which is exactly the axis `iALM`, `nuPI` and `PBM` sit on.

Three choices worth stating because they change the numbers:

- **The density denominator runs over gated groups only.** Qwen2.5-0.5B's tied
  151936 x 896 embedding is 27.5 % of the model and cannot be gated, so a whole-model
  denominator would make every target below 0.275 infeasible by construction. Their Fig. 1
  reports whole-model parameter percentage in a separate panel for the same reason. This is
  the choice that makes density numbers incomparable between papers, so it is stated rather
  than assumed.
- **Gate parameters get their own learning rate** (`--gate-lr`, default 1e-2 against the
  model's 1e-4). Adam moves a parameter by roughly its learning rate per step whatever the
  gradient scale, and carrying `log_alpha` from a 95 %-open initialisation to a 30 %-density
  solution is a move of about 3.8 — at the model's step size that is ~38k steps, so a shared
  step size would turn this into a measurement of the learning rate. The reference uses
  separate optimizers for weights and gates too.
- **Timing is against the *gated* model, not the plain one.** `scaling.py`'s control is
  `adam` with gates attached and no constraint, so the difference is the dual layer; the
  gates are a property of the sparsity formulation, not of any optimizer. Same control
  design as E2a. Phases are timed with `torch.cuda.Event` around an explicit
  `synchronize()` — the per-epoch `time` columns elsewhere in this repo wrap asynchronous
  launches and so measure launch time.

**What E3 does not show: correctness under data parallelism.** The density constraint is a
closed form in the gate parameters with no data in it, and DDP keeps those parameters
identical on every rank, so `all_reduce(c, AVG)` averages identical values and cannot be
caught being wrong. The collective still costs bytes and still synchronises; it just is not
load-bearing here. That claim belongs to [e2/b_parallel.py](e2/b_parallel.py), whose
constraints are sample averages over a shard. The deciding property is whether the
constraint is a function of the *data*.

Running it needs two things this repository does not carry: a token shard from
[e3/prepare_data.py](e3/prepare_data.py) (login node — compute nodes have no network) and
`transformers` in a venv layered on the cluster's PyTorch module, since no EasyBuild module
provides it. [e3/sbatch_e3.sh](e3/sbatch_e3.sh) does both. Everything below the model runs
without either: `--model stub` uses [problems/tiny_lm.py](problems/tiny_lm.py), a Llama/Qwen-shaped
fixture, so the gates, constraints, duals, distributed path and artifact writing are all
exercisable on CPU.

## Layout

```
paper/
  README.md                 this file
  _harness.py               seeding, float64, CSV/JSON/Markdown writers, PDF figures, Checks
  problems/
    __init__.py             the Problem dataclass, including the derived primal step size
    qp.py                   convex and nonconvex QPs with exact (x*, y*)
    svm.py                  hard-margin SVM on separable Iris, exact active-set reference
    nonsmooth.py            the ten Karmitsa problems + transcription verification
    fairness.py             ACSIncome + Dutch census, pairwise / aggregate fairret constraints
    sparse_lm.py            hard-concrete L0 gates + the parameter-counted density constraints
    tiny_lm.py              a Llama/Qwen-shaped decoder, so E3 runs without `transformers`
    tokens.py               uint16/uint32 token shards + a rank-sharded block sampler
  e0/
    a_multipliers.py  b_nonopt.py  d_distributed.py  run_all.sh
  e2/
    a_fairness.py  b_parallel.py  run_all.sh
  e3/
    run_llm.py              one configuration, end to end, under torchrun
    sweep.py  scaling.py    the two drivers
    prepare_data.py         login-node one-off: FineWeb-Edu -> tokens.bin
    sbatch_e3.sh  run_all.sh
  results/e0a/ e0b/ e0d/    *.csv (raw), *.md + *.json (tables), *.pdf (figures)
  results/e2a/ e2b/ e3/
```

Figure style comes from [benchmark/new_bench/plotting/plot_style.py](../benchmark/new_bench/plotting/plot_style.py),
loaded by file path because that tree is not an importable package. It is the
repo's declared single style source, and duplicating its rcParams here would let
the two drift.

## Not here yet

**E1** (synthetic benchmark) is blocked on choosing a problem source. S2MPJ was
rejected after probing all 1104 of its Python problems: no PyTorch/autograd API
(numpy `fgx`/`cJx`, so every oracle call needs a hand-plumbed
`torch.autograd.Function`), not on PyPI, **no LICENSE file** so it cannot be vendored
into a reproducibility artifact, finite variable bounds and range constraints against
this package's bounds-free template, and >20 s constructors on several problems.

**E4** (nonsmooth reformulations) is not started. `problems/` is shared by design.
