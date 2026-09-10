# `algorithm/` — one file per dual optimizer configuration

`dual` is a `hydra.utils.instantiate` spec with `_partial_: true`; the adapter closes it
with `m=` (and `device=`/`process_group=` for E3-CIFAR). It is the **single source of
truth for every dual hyperparameter** on the E2a and E3-CIFAR paths, including the step
size — which is called `lr` on `ALM`, `beta` on `iALM`, `ki`/`kp` on `nuPI` and
`penalty_mult` on `PBM`, so there is no uniform "dual lr" key to sweep.

`method` names the entry in `run_llm.METHODS` / `run_cifar.METHODS`. The **E3-LLM adapter
reads only that field** and ignores `dual`, because it drives `run_llm.main()` through
argv (see `conf/experiment/e3_llm.yaml`); `method: null` means the algorithm has no E3-LLM
counterpart and the adapter says so rather than silently substituting one.

`primal.lr` is E2a only. E3 has its own `lr`/`gate_lr` split in the experiment config,
because it gives the weights SGD and the gates Adam at a much larger step.

Defaults are transcribed from the registries they mirror — `a_fairness.METHODS`
(`DUAL_LR = 0.05`, `PRIMAL_LR = 1e-3`) and `run_llm.METHODS` — so an untouched config
reproduces the committed baseline rather than a third, unrelated setting.
