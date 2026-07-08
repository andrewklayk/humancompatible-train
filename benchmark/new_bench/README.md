# new_bench — dual_optim benchmark (Hydra multirun)

One job = ONE `(data, task, algorithm, hyperparameters, seed)`. Sweeping, selection,
and plotting are separate, re-runnable steps that read the saved results. Run
everything from this directory.

**The focus is the `opt` (optimization / KKT) experiments** — the whole dataset is the
training set and we measure how closely each method approaches a KKT point. The `ml`
(cross-validated train/val/test) view still works and is documented near the end.

## Layout

| Path | Responsibility |
|------|----------------|
| `run.py` | Hydra entrypoint: data → task → algorithm → train → write raw histories |
| `train.py` | the single training loop + `evaluate_optimality` (full-batch KKT metrics) |
| `algorithms.py` | update strategies (plain / primal_dual / switching) from `cfg.algorithm` |
| `tasks.py`, `data.py`, `constraints.py`, `models.py`, `_loaders.py` | task / data / model plumbing |
| `tune.py`, `search_spaces.py` | **Optuna tuning** (sampled search, feasibility-first objective) |
| `conf/sweep/` | per-algorithm manual grids (basic sweeper) |
| `scripts/E1_opt_*.sh` | per-algorithm `opt` grid launchers + `E1_opt_select.sh` |
| `scripts/run_all_opt.sh` | chain the `opt` sweeps as an sbatch dependency chain |
| `scripts/tune.sh`, `scripts/_env.sh` | Optuna launcher + shared env/defaults |
| `aggregate.py`, `select_best.py` | multirun tree → seed-averaged aggregates → best-config picks |
| `plotting/` | figures over `selection/…/aggregated/` (`plot_kkt` for opt; pareto / cdf / fair) |

## Run one config

```bash
python run.py data=income task=folktables_positive_rate_pair algorithm=pbm approach=opt
```

Writes `train.csv`, `opt.csv` (KKT metrics, `opt` mode), `run_meta.json` to the job's
output dir. Algorithms: `adam`, `pbm`, `pbm_logscaled`, `alm_proj`, `alm_max`, `ssg`.
The primal Adam (and ssg's dual Adam) default to `weight_decay=0.01`.

## The `opt` approach

`approach=opt` is a pure stochastic-optimization view: the **entire dataset is the
training set**, with no val/test split and no folds — `init_seed` (model init + batch
order) is the only randomness axis. Each epoch also logs **full-batch KKT optimality
metrics** to `opt.csv`, computed at the frozen end-of-epoch iterate over the whole
training set (*not* an epoch-mean of minibatch values, which would carry a
gradient-noise floor):

| column | meaning |
|--------|---------|
| `f` | full-batch objective (loss) at the iterate — logged for **every** method |
| `grad_norm` | ‖∇ₓ L‖ with `L = f + λᵀ(c−b)` (stationarity residual; `‖∇ₓ f‖` for ssg/adam) |
| `L` | Lagrangian value (primal-dual methods) |
| `max_viol` | `maxⱼ(cⱼ−b)` — primal feasibility (≤0 feasible) |
| `compl` | `Σⱼ abs(λⱼ(cⱼ−b))` — complementarity (primal-dual) |
| `lambda_j` | dual variables (primal-dual) |

For **tabular** tasks this is exact (the whole set fits one forward+backward). For
**image** tasks it uses a fixed, class-balanced subsample of size `opt_eval_size`
(default 10000; lower it if the backward OOMs). `ml` and `opt` runs must be
**aggregated separately** (`aggregate.py --approach`, separate `--out`).

## Sweep the `opt` experiments

Two ways to explore hyperparameters — both write per-config runs into `multirun/` in
the same layout, so `aggregate.py` / `select_best.py` / `plot_kkt` consume either.

### IMPORTANT

Before running any experiment for the first time, make a dummy training run to load the data (run from interactive node after loading modules):

```bash
python run.py data=cifar10 task=cifar10_loss algorithm=adam approach=opt n_epochs=1
```

### Experiments

```bash
E1: DATA=income_norm TASK=weight_norm
E2: DATA=income TASK=folktables_positive_rate_vec
E3: DATA=income TASK=folktables_positive_rate_pair
E4: DATA=dutch TASK=dutch_positive_rate_pair
E5: DATA=cifar10 TASK=cifar10_loss LAUNCHER=slurm_h200fast
E6: DATA=cifar100 TASK=cifar100_loss LAUNCHER=slurm_h200
```

### 1. Manual grid (basic sweeper)

One driver per algorithm, each independently (re-)runnable. The grid lives in
`conf/sweep/<algo>.yaml` (`hydra.sweeper.params`, full cartesian product). Only
`init_seed` is swept (no fold loop):

```bash
DATA=income TASK=folktables_positive_rate_pair sbatch scripts/E1_opt_pbm.sh            # also: adam, alm_proj, ssg
bash scripts/E1_opt_select.sh         # aggregate + select, after any subset finishes
```
---
Chain all of them as an sbatch **dependency chain** (each sweep starts only after the
previous finishes) + a final aggregate/select job:

```bash
DATA=... TASK=... sbatch scripts/run_all_opt.sh
```

`ALGOS` (order/set, default `adam alm_proj ssg pbm`), `SELECT=0`
to skip selection.

### 2. Optuna tuning (sampled search)

`tune.py` replaces the exhaustive grid with a TPE search and is the recommended way to
tune — it covers **all six** algorithms (including `alm_max`, `pbm_logscaled`, which
have no grid driver). Search spaces are in `search_spaces.py` (continuous/log ranges
for lrs/penalties, categorical for structural switches).

```bash
ALGO=pbm bash scripts/tune.sh                              # 8 workers x 25 trials on Slurm
LAUNCHER=local ALGO=pbm N_TRIALS=50 bash scripts/tune.sh   # single local worker
```

- **Objective — feasibility-first** (constrained methods): minimize the full-batch
  objective `f`, with an infinite penalty for infeasibility:
  `score = f if max_viol ≤ --feas-tol (default 1e-3) else +inf`. Both `f` and
  `max_viol` are read from `opt.csv` (same frozen iterate), tail-averaged over `--tail`
  epochs and averaged over `--init-seeds`. `adam` (unconstrained reference) is tuned on
  `f` alone. `+inf` is stored as a large finite penalty for sampler stability.
- **Parallelism** — N workers share ONE Optuna study via a `JournalStorage` file, so
  `tune.sh` submits a **fixed small** job array (not a giant grid array) → no
  `MaxArraySize` / `QOSMaxSubmitJobPerUser` limits. Sampler uses `constant_liar` so
  concurrent workers diversify.
- **Outputs** — every trial's curves are written to
  `multirun/<task>/<algo>/opt/trial<n>_init<seed>/` (full parity with the grid), and the
  study records each trial's params + scalar in the journal file. A dependent finalize
  job writes `best_params.json`. Then run `bash scripts/E1_opt_select.sh`.

Knobs: `ALGO`, `N_WORKERS` (concurrency/adaptivity trade-off; keep ≪ total trials),
`N_TRIALS` (per worker), `N_EPOCHS`, `TAIL`, `DATA`, `TASK`, `INIT_SEEDS`, and
`tune.py --feas-tol`.

## Slurm

`conf/hydra/launcher/slurm_gpu.yaml` (GPU) / `slurm.yaml` (CPU) route each trial to a
Slurm array task via submitit (partition, mem, `array_parallelism` = concurrency cap).
Two layers, on possibly different partitions:

- **Children** (the actual training array) get their partition/`gres`/mem from the
  launcher yaml, submitted by submitit.
- **Driver** = the `E1_opt_*.sh` / `tune.py` process. It **blocks** while its array
  runs, so when you `sbatch` it (or `run_all_opt.sh`), give it a **long walltime on a
  non-fast partition** (`amdgpufast` would kill it mid-sweep) and no GPU. `run_all_opt.sh`
  sets this via `PARTITION`/`TIME`.

Hygiene / gotchas (handled in `scripts/_env.sh`):

- **Launch grid drivers with `bash` on the login node, or `sbatch` them on a long
  partition** — submitit submits the array itself; don't nest submissions carelessly.
- **`SLURM_MEM_*` leak** — a driver's `--mem` exports `SLURM_MEM_PER_NODE` into the
  submitit children, colliding with the launcher's `--mem-per-cpu`
  (`srun: … mutually exclusive`). `_env.sh` unsets `SLURM_MEM_PER_NODE/_PER_CPU/_PER_GPU`
  before any `run.py -m`.
- **MaxArraySize / QOS submit limit** — each `run.py -m` is ONE array; the drivers peel
  `init_seed` (and `lr`) into an outer loop so each array stays under the cluster's
  `MaxArraySize`, and submitit blocks per chunk so the queue never overfills. Optuna
  tuning sidesteps both by using a fixed worker pool instead.
- Launch from a **shared filesystem**; **pre-download datasets once** on the login node.
- Chain jobs with dependencies (note: sbatch flags must come **before** the script):
  `sbatch --dependency=afterok:$JID scripts/E1_opt_ssg.sh` — `run_all_opt.sh` does this
  for you with `--parsable`.

## Aggregate + select (two stages)

Split into two re-runnable scripts sharing the `aggregated/` files as the single source
of truth (also read by the plots):

```bash
# stage 1 — raw multirun -> per-config seed-averaged curves (opt into its own dir)
python aggregate.py   --runs multirun/ --out selection/opt --approach opt
# stage 2 — aggregated curves -> best config per cell (cheap; re-run at any --tols)
python select_best.py --agg selection/opt/aggregated --out selection/opt --tols 1.0,1.1,1.25 --tail 5
```

`scripts/E1_opt_select.sh` runs both for you. (`ml` uses `--approach ml` into
`selection/`.)

**`aggregate.py`** — per `(task, data, algorithm)` cell, matches configs across the
`init_seed` (and `fold`, in `ml`) grid and writes per-epoch curves for **every split**,
including each constraint `c_i`, the max violation, and the KKT metrics (`f`,
`grad_norm`, `L`, `compl`, `max_viol`, `lambda_i`) with `_std_fold`/`_std_init` variance
components:

```text
aggregated/<cell>.csv    # all curves, long: one row per (config, split, epoch)
aggregated/<cell>.json   # per-config metadata + hyperparameters (curves live in the CSV)
```

**`select_best.py`** — reads those JSONs (never the raw runs), collapses each config's
selection-split curve over a window (mean of the last `--tail` epochs, or `--rolling`),
and among configs feasible at `bound·mult` takes the min-loss one, once per `--tols`
slack level. The windowed `collapse` is imported by the plotting backend so selection
and plots always agree. In `opt` mode selection is on the **train** curve.

## Plotting

Reads `selection/opt/aggregated/` — **run `select_best.py` (or `E1_opt_select.sh`)
first**. From `plotting/` (backend: `prepare_results_plotting.py`):

**KKT closeness (`opt`).** `plot_kkt.py` reads the `opt` split:

```bash
cd plotting
python plot_kkt.py --agg ../selection/opt/aggregated \
    --task folktables_positive_rate_pair --data income --mode cdf --metric residual
```

- `--mode`: `cdf` (default) / `pdf` (closeness over configs), `scatter`, `conv`
  (metric-vs-epoch), `all` (2×2), `duals` (λⱼ vs epoch per constrained method).
- `--metric`: `residual` (`‖∇L‖ + relu(max_viol) + |compl|`), `grad_norm`, `max_viol`,
  `compl`, or `objective` (the loss `f`, read from the train split).
- `--metric objective` additionally drops **infeasible** configs (final
  `max_viol > --feas-tol`, default 0.0) so a low loss can't come from ignoring the
  constraints. `--linear` switches off the log axis; `--tail` re-collapses at plot time.

The backend also exposes config filtering: `list_configs(spec, method, where=...)`
keeps only configs matching hyperparameter constraints (exact value or a predicate),
and `config_params(...)` returns a config's flattened hyperparameters.

**Fairness / ML views.** `plot_pareto.py`, `plot_cdf.py`, `plot_fair.py` operate on the
`ml` aggregates (`--split {train,val,test}`, `--tol` slack, `--companion`).

## The `ml` approach (cross-validated)

`approach=ml` (the default) is the machine-learning view: a fixed stratified **test**
set is held out, the dev set is K-folded, and randomness is split into two axes —
**`fold`** (data variance) and **`init_seed`** (optimization variance). `select_best.py`
aggregates over the full `fold × init_seed` grid and reports a variance decomposition
(`*_std_fold` vs `*_std_init`); test metrics are read at the val-selected epoch, never
used for selection. Launchers: `scripts/E1_{pbm,alm_proj,alm_max,ssg,adam}.sh` +
`E1_select.sh`. Defaults (`scripts/_env.sh`): `n_folds=5`, `init_seed` over 3 values,
`cv_seed=0`. Stratification is by sensitive group (class for cifar).

## Reference

- **data:** `income`, `income_sex`, `income_all`, `dutch`, `income_norm`, `cifar10`, `cifar100`
- **task:** `folktables_positive_rate_pair`, `dutch_positive_rate_pair`, `folktables_positive_rate_vec`, `weight_norm`, `cifar10_loss`, `cifar100_loss`
- **algorithm:** `adam`, `pbm`, `pbm_logscaled`, `alm_proj`, `alm_max`, `ssg`
  (grid `opt` drivers exist for adam/alm_proj/pbm/ssg; `tune.py` covers all six)
- **model** (`task.model`): `mlp` (tabular), `conv` (small ConvNet), `resnet`
  (torchvision resnet18, ImageNet stem — matches the original benchmark), `resnet_cifar`
  (resnet18 with a 3×3/stride-1 stem and no maxpool, for 32×32 inputs). `cifar100_loss`
  defaults to `resnet`; override per run, e.g. `task.model=resnet_cifar`.

The training loop is verified bit-exact against the old `run_train` on folktables.
(ssg's primal is Moreau-wrapped here — a fix vs the old `utils.py`.)
