# new_bench — dual_optim benchmark (Hydra multirun)

One job = ONE `(data, task, algorithm, hyperparameters, seed)`. Hydra multirun
produces the sweep; selection and plotting are separate, re-runnable steps that
read the saved results. Run everything from this directory.

## Layout

| Path | Responsibility |
|------|----------------|
| `run.py` | Hydra entrypoint: data → task → algorithm → train → write raw histories |
| `train.py` | the single training loop + eval |
| `algorithms.py` | update strategies (plain / primal_dual / switching) from `cfg.algorithm` |
| `tasks.py`, `data.py`, `constraints.py`, `models.py`, `_loaders.py` | task / data / model plumbing |
| `conf/sweep/` | per-algorithm search spaces |
| `scripts/E1_*.sh` | per-algorithm sweep launchers + `E1_select.sh` + shared `_env.sh` |
| `select_best.py` | scan a multirun tree → seed-averaged aggregates + best-config picks |
| `plotting/` | figures over `selection/aggregated/` (pareto / cdf / fair) |

## Run one config

```bash
python run.py data=income task=folktables_positive_rate_pair algorithm=pbm
```

Writes `train.csv`, `val.csv`, `test.csv`, `run_meta.json` to the job's output dir.

## Sweep (E1)

One script per algorithm, each independently (re-)runnable. Each runs a **manual
grid** via Hydra's built-in **basic sweeper** — the grid lives in
`conf/sweep/<algo>.yaml` (`hydra.sweeper.params`, full cartesian product); `adam`
is the unconstrained reference on a small lr grid.

```bash
bash scripts/E1_pbm.sh          # also: alm_proj, alm_max, ssg, adam
bash scripts/E1_select.sh       # selection, after any subset finishes
```

Defaults live in `scripts/_env.sh` and are overridable inline:

```bash
LAUNCHER=slurm INIT_SEEDS="0 1 2" N_FOLDS=5 bash scripts/E1_pbm.sh
LAUNCHER=local INIT_SEEDS=0 N_FOLDS=2 bash scripts/E1_pbm.sh    # quick smoke test
```

Key knobs: `DATA`, `TASK`, `INIT_SEEDS` / `N_FOLDS` / `CV_SEED` (see
Cross-validation below), `LAUNCHER` (`slurm` | `slurm_gpu` | `local`). To resize a
grid, edit the `choice(...)` lists in `conf/sweep/<algo>.yaml`.

Two things that are easy to get wrong, both handled by the scripts:

- **(fold, init_seed) are looped, not multirun axes** (for the constrained methods).
  Each `(fold, init_seed)` is a separate basic-sweeper run over the SAME grid, so
  every run produces identical configs and `select_best.py` matches them across the
  CV grid by hyperparameter signature. (`adam` instead folds fold/init into the
  cartesian multirun directly.)
- **Launch on the login node with `bash`, not `sbatch`.** The submitit launcher
  submits the array itself; `sbatch`-ing the driver nests submissions.

## Slurm

`LAUNCHER=slurm` (CPU, fairness tasks) / `slurm_gpu` (cifar) routes each trial to a
Slurm array task via submitit. Edit resources in `conf/hydra/launcher/slurm*.yaml`
(partition, mem, `array_parallelism` = real concurrency cap, `setup:` for env).
Hygiene:

- Launch from a **shared filesystem** — `multirun/`, `selection/`, `./data` are
  relative to the launch dir and every node must see them.
- **Pre-download datasets once** on the login node (`download=True` racing across
  array tasks can corrupt `./data`).

## Cross-validation

Randomness is split into two independent axes (config fields, swept by the scripts):

- **`fold` ∈ `0..n_folds-1`** — the **data** axis. A fixed, stratified **test** set is
  held out *before* K-folding (`cv_seed`, `test_size`); the remaining *dev* set is
  partitioned by `StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)`,
  and `fold` selects which fold is validation. Stratification is by sensitive group
  (class for cifar), so every group appears in the test set and each fold — required
  by `BalancedBatchSampler` and the per-group constraints. The `StandardScaler` is fit
  on each fold's train only.
- **`init_seed`** — the **optimization** axis: model-weight init and batch order.

`cv_seed`/`fold`/`test_size` fix the partition; `init_seed` is the only thing that
varies the model given the data. `select_best.py` aggregates each config over the full
`fold × init_seed` grid and reports a **variance decomposition**: `*_std_init`
(optimization variance, across init seeds within a fold) vs `*_std_fold` (data
variance, across folds). The held-out **test** metrics (`test_loss_*`, `test_maxc_*`)
are read at the val-selected epoch — the test set is never used for selection.

Defaults: `n_folds=5`, `init_seed` swept over 3 values, `cv_seed=0`. Cost per method
≈ `n_configs × n_folds × n_init_seeds`.

## Two approaches: `ml` vs `opt`

Set `approach=` on any run (default `ml`):

- **`ml`** — the machine-learning view above: K-fold CV, train/val/test, val-based
  selection.
- **`opt`** — a pure stochastic-optimization view: the **entire dataset is the
  training set**, no val/test split and no folds (`init_seed` is the only randomness
  axis). Selection falls back to the **train** curve. Run the `scripts/E1_opt_*.sh`
  launchers (one `init_seed` multirun each, no fold loop) then `scripts/E1_opt_select.sh`.

In `opt` mode, each epoch also logs **full-batch KKT optimality metrics** to `opt.csv`
(at the frozen end-of-epoch iterate, on the whole training set — *not* an epoch-mean
of minibatch values, which would carry a gradient-noise floor):

| column | meaning |
|--------|---------|
| `grad_norm` | ‖∇ₓ L‖ with `L = f + λᵀ(c−b)` (stationarity residual; `‖∇f‖` for ssg/adam) |
| `L` | Lagrangian value (primal-dual methods) |
| `max_viol` | `maxⱼ(cⱼ−b)` — primal feasibility (≤0 feasible) |
| `compl` | `Σⱼ abs(λⱼ(cⱼ−b))` — complementarity (primal-dual) |
| `lambda_j` | dual variables (primal-dual) |

For **tabular** tasks this is exact (the whole set fits one forward+backward). For
**image** tasks it is computed on a fixed, class-balanced subsample of size
`opt_eval_size` (default 10000; lower it if the backward OOMs) — exact for that
subsampled problem, and CIFAR is class-balanced so the subsample is representative.
`ml` and `opt` runs must be selected **separately** (`select_best.py --approach`,
separate `--out`); mixing them in one selection cell errors out.

## Select best configs

```bash
python select_best.py --runs multirun/ --out selection/ --approach ml --tols 1.0,1.1,1.25 --tail 5
python select_best.py --runs multirun/ --out selection/opt --approach opt --tail 5   # opt cells
```

Per `(task, data, algorithm)` cell, matches configs across the `fold × init_seed`
grid and writes per-epoch curves for **every split**, including each constraint `c_i`,
the max violation, and the `_std_fold`/`_std_init` variance components. Selection: each config's val curve is collapsed over a window
(mean of the last `--tail` epochs, or `--rolling` for the rolling-`tail` argmin-loss
epoch); among configs feasible at `bound·mult` the min-loss one wins, once per
`--tols` slack level (`adam`/`filter=none` → plain argmin loss). Output under `--out`:

```text
aggregated/<cell>__cfgNNN.json   # per config: hyperparameters + all-split per-constraint curves
aggregated/<cell>__cfgNNN.csv    # the val curve, for eyeballing
best_<cell>__tol<mult>.json      # selected config per (cell, slack)
best_summary.csv                 # one row per (cell, slack)
```

Selection is auditable and re-runnable without retraining. The windowed `collapse`
is shared with the plotting backend, so selection and plots always agree.

## Plotting

Reads `selection/aggregated/` — **run `select_best.py` first**. From `plotting/`:

```bash
cd plotting
python plot_pareto.py --task folktables_positive_rate_pair --data income --bound 0.1 --pareto
python plot_cdf.py    --task folktables_positive_rate_pair --data income --bound 0.1
python plot_fair.py   --task folktables_positive_rate_pair --data income --bound 0.1 --tol 1.0 --companion val
```

`--agg` defaults to `../selection/aggregated`. `plot_cdf`/`plot_pareto` take
`--split {train,val,test}` (single split) and `--tail` to re-collapse the stored
curves at plot time. `plot_fair` plots train + a `--companion {val,test}` split
(default `test`) and reads the winner from `best_*.json` (pick the slack with
`--tol`); its renderer is reused from `../../plotting/`.

**KKT closeness (`opt` approach only).** `plot_kkt.py` reads the `opt` split and plots
the composite KKT residual `r = ‖∇L‖ + max(0,max_viol) + abs(compl)` (each config
collapsed to the mean of its last `--tail` epochs):

```bash
python plot_kkt.py --agg ../selection/opt/aggregated \
    --task folktables_positive_rate_pair --data income --mode cdf --metric residual
```

`--mode cdf` (default) / `pdf` give the closeness-over-configurations view (fraction
of configs reaching each residual / its histogram); `scatter` is one point per config;
`conv` is residual-vs-epoch (faint per config, bold best); `all` is the 2×2 grid.
`--metric {residual,grad_norm,max_viol,compl}` isolates a single KKT component;
`--linear` switches off the default log metric axis.

## Reference

- **data:** `income`, `income_sex`, `income_all`, `dutch`, `income_norm`, `cifar10`, `cifar100`
- **task:** `folktables_positive_rate_pair`, `dutch_positive_rate_pair`, `folktables_positive_rate_vec`, `weight_norm`, `cifar10_loss`
- **algorithm:** `adam`, `pbm`, `alm_proj`, `alm_max`, `ssg`

The training loop is verified bit-exact against the old `run_train` for all five
algorithms on folktables. (ssg's primal is Moreau-wrapped here — a fix vs the old
`utils.py`, which left it a plain optimizer.)
