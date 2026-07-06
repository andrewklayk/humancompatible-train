"""Optuna search spaces for the opt-experiment tuning (tune.py).

One function per algorithm returns a dict ``{dotted_cfg_key: sampled_value}`` that
tune.py turns into Hydra config overrides. The design mirrors conf/sweep/<algo>.yaml
but replaces the exhaustive grid with distributions:

  * continuous knobs (learning rates, penalties, dual step size) -> sampled over a
    range, log-uniform where they span orders of magnitude;
  * structural switches (booleans, integer process length, Moreau ``mu`` levels, the
    PBM ``penalty_range`` pairs) stay CATEGORICAL so their discrete meaning (e.g.
    ``mu=0`` = Moreau off) is preserved -- continuous sampling would never hit them.

Ranges are the sweep grids' span, widened slightly for the continuous axes.
"""

# PBM penalty_range pairs (lists aren't hashable, so key the categorical by a string).
_PBM_RANGES = {"[0.1, 1.0]": [0.1, 1.0], "[0.01, 2.0]": [0.01, 2.0]}


def _adam(trial):
    return {"algorithm.primal.lr": trial.suggest_float("primal_lr", 1e-3, 5e-2, log=True)}


def _pbm(trial):
    return {
        "algorithm.primal.lr": trial.suggest_float("primal_lr", 1e-3, 5e-2, log=True),
        "algorithm.dual.penalty_mult": trial.suggest_float("penalty_mult", 0.0, 1.0),
        "algorithm.dual.penalty_range":
            _PBM_RANGES[trial.suggest_categorical("penalty_range", list(_PBM_RANGES))],
        "algorithm.dual.primal_update_process_length": trial.suggest_int("pupl", 1, 3),
        "algorithm.dual.gamma_annealing": trial.suggest_categorical("gamma_annealing", [True, False]),
        "algorithm.moreau.mu": trial.suggest_categorical("moreau_mu", [0.0, 2.0]),
    }


def _pbm_logscaled(trial):
    return {
        "algorithm.primal.lr": trial.suggest_float("primal_lr", 1e-3, 5e-2, log=True),
        "algorithm.dual.penalty_mult": trial.suggest_float("penalty_mult", 0.0, 1.0),
        "algorithm.moreau.mu": trial.suggest_categorical("moreau_mu", [0.0, 1.0]),
        # the mirror-map dual step size is the distinguishing knob of this variant.
        "algorithm.dual.logscaled_dual_step_size": trial.suggest_float("dual_step", 1e-2, 5e-1, log=True),
    }


def _alm_max(trial):
    return {
        "algorithm.primal.lr": trial.suggest_float("primal_lr", 1e-3, 5e-2, log=True),
        "algorithm.dual.lr": trial.suggest_float("dual_lr", 1e-3, 5e-2, log=True),
        "algorithm.dual.penalty": trial.suggest_float("penalty", 1e-1, 1e1, log=True),
        "algorithm.moreau.mu": trial.suggest_categorical("moreau_mu", [0.0, 1.0, 2.0]),
    }


def _alm_proj(trial):
    return {
        "algorithm.primal.lr": trial.suggest_float("primal_lr", 1e-3, 5e-2, log=True),
        "algorithm.dual.lr": trial.suggest_float("dual_lr", 1e-3, 5e-2, log=True),
        "algorithm.dual.penalty": trial.suggest_float("penalty", 0.0, 2.0),
        "algorithm.moreau.mu": trial.suggest_categorical("moreau_mu", [0.0, 1.0]),
    }


def _ssg(trial):
    return {
        "algorithm.primal.lr": trial.suggest_float("primal_lr", 1e-3, 5e-2, log=True),
        "algorithm.dual.lr": trial.suggest_float("dual_lr", 1e-3, 5e-2, log=True),
        "algorithm.moreau.mu": trial.suggest_categorical("moreau_mu", [0.0, 1.0, 2.0]),
    }


SEARCH_SPACES = {
    "adam": _adam,
    "pbm": _pbm,
    "pbm_logscaled": _pbm_logscaled,
    "alm_max": _alm_max,
    "alm_proj": _alm_proj,
    "ssg": _ssg,
}


def suggest(trial, algo):
    """Sample one config for ``algo``; returns {dotted_cfg_key: value}."""
    if algo not in SEARCH_SPACES:
        raise ValueError(f"No search space for '{algo}'. Known: {sorted(SEARCH_SPACES)}")
    return SEARCH_SPACES[algo](trial)
