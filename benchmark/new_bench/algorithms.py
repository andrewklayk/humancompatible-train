"""Algorithm layer.

Replaces the old ``mode`` string ('hc'/'sw'/'torch') + ``isinstance(dual, PBM)``
dispatch and the three near-identical training loops. Each algorithm is built
from ``cfg.algorithm`` into an ``Algorithm`` whose ``.step()`` performs ONE
per-batch update. The single training loop in ``train.py`` calls ``.step()`` and
knows nothing about which algorithm it runs.

Three update strategies cover all five algorithms:
  * plain        -> adam            (loss.backward(); primal.step())
  * primal_dual  -> pbm, alm_proj, alm_max
                    (lgr = dual.forward_update(loss, c_eq); lgr.backward(); primal.step())
  * switching    -> ssg             (step dual on max violation, else primal on loss)
"""
from dataclasses import dataclass
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from humancompatible.train.dual_optim import MoreauEnvelope


@dataclass
class Algorithm:
    name: str
    primal: object                       # primal optimizer (possibly Moreau-wrapped)
    dual: Optional[object]               # dual optimizer / second primal (ssg), or None
    updater: str                         # 'plain' | 'primal_dual' | 'switching'
    constraints_to_eq: bool
    select_filter: str                   # used downstream by select_best.py (saved in config)
    passes_loss_to_constraints: bool     # whether the unreduced loss is fed to the constraint fn
    constraint_tol: float = 0.0          # switching method only
    grad_clip: Optional[float] = None    # max grad-norm; None disables clipping

    def zero_grad(self):
        self.primal.zero_grad()

    def _clip(self):
        """Clip the model's gradient norm in place; no-op when disabled. Guards
        against the divergence that turns the loss into NaN on high-LR / annealed
        penalty-barrier runs (the poisoned weights never recover). The model
        params are the primal optimizer's param groups (Moreau delegates via
        __getattr__), the same params the dual steps in ``switching``."""
        if self.grad_clip is not None:
            params = [p for g in self.primal.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

    def step(self, loss_mean, constraints_bounded_eq):
        """One optimization step. ``loss_mean`` is the scalar (mean) loss;
        ``constraints_bounded_eq`` is the (c - bound) [or max(c-bound,0)] tensor.
        Assumes ``zero_grad()`` and the forward pass already happened."""
        if self.updater == "plain":
            loss_mean.backward()
            self._clip()
            self.primal.step()
        elif self.updater == "primal_dual":
            lgr = self.dual.forward_update(loss_mean, constraints_bounded_eq)
            lgr.backward()
            self._clip()
            self.primal.step()
        elif self.updater == "switching":
            max_c = max(constraints_bounded_eq)
            if max_c > self.constraint_tol:
                max_c.backward()
                self._clip()
                self.dual.step()
            else:
                loss_mean.backward()
                self._clip()
                self.primal.step()
        else:
            raise ValueError(f"Unknown updater '{self.updater}'.")


def _is_pbm(dual_cfg) -> bool:
    return str(dual_cfg["_target_"]).split(".")[-1] == "PBM"


def build_algorithm(cfg_algo, model, m, epoch_length) -> Algorithm:
    """Construct the optimizers and the Algorithm wrapper from ``cfg.algorithm``."""
    updater = cfg_algo["updater"]
    name = cfg_algo["name"]
    grad_clip = cfg_algo.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    # ----- primal optimizer (e.g. Adam), built from a partial _target_ -----
    primal_opt = instantiate(cfg_algo["primal"])(model.parameters())

    if updater == "plain":
        # adam: bare primal, no dual, no Moreau envelope.
        return Algorithm(
            name=name, primal=primal_opt, dual=None, updater=updater,
            constraints_to_eq=bool(cfg_algo.get("constraints_to_eq", False)),
            select_filter=cfg_algo.get("select_filter", "none"),
            passes_loss_to_constraints=True,
            grad_clip=grad_clip,
        )

    # Moreau-wrapped primal for both primal_dual and switching.
    moreau_kwargs = OmegaConf.to_container(cfg_algo["moreau"], resolve=True) if "moreau" in cfg_algo else {}

    if updater == "switching":
        # ssg: both the primal (loss) and dual (constraint) steps are Moreau-wrapped,
        # on the same model params, with independent optimizer configs (the grid sweeps
        # primal.lr and dual.lr separately). Falls back to the primal config if no dual.
        dual_opt_cfg = cfg_algo.get("dual", cfg_algo["primal"])
        primal = MoreauEnvelope(primal_opt, **moreau_kwargs)
        dual = MoreauEnvelope(instantiate(dual_opt_cfg)(model.parameters()), **moreau_kwargs)
        return Algorithm(
            name=name, primal=primal, dual=dual, updater=updater,
            constraints_to_eq=False,
            select_filter=cfg_algo.get("select_filter", "upper"),
            passes_loss_to_constraints=False,  # original sw loop feeds loss=None to the constraint fn
            constraint_tol=float(cfg_algo.get("constraint_tol", 0.0)),
            grad_clip=grad_clip,
        )

    if updater == "primal_dual":
        dual_cfg = cfg_algo["dual"]
        dual_partial = instantiate(dual_cfg)
        if _is_pbm(dual_cfg):
            # PBM needs epoch_length; Moreau's process length must match the dual's.
            pupl = int(dual_cfg.get("primal_update_process_length", 1))
            moreau_kwargs["primal_update_process_length"] = pupl
            dual = dual_partial(m=m, epoch_length=epoch_length)
        else:
            dual = dual_partial(m=m)
        primal = MoreauEnvelope(primal_opt, **moreau_kwargs)
        return Algorithm(
            name=name, primal=primal, dual=dual, updater=updater,
            constraints_to_eq=bool(cfg_algo.get("constraints_to_eq", False)),
            select_filter=cfg_algo.get("select_filter", "upper"),
            passes_loss_to_constraints=True,
            grad_clip=grad_clip,
        )

    raise ValueError(f"Unknown updater '{updater}' for algorithm '{name}'.")
