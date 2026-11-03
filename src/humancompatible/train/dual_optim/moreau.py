import torch
from torch.optim import Optimizer


class MoreauEnvelope(Optimizer):
    r"""
    A wrapper over a PyTorch `Optimizer` that adds the gradient of the Moreau-envelope
    smoothing term :math:`\frac{\mu}{2} \|\theta - z\|^2` before each primal step.

    The prox centre :math:`z` is an EMA of the iterates, updated from the *pre-step*
    iterate :math:`\theta_t`: :math:`z \leftarrow (1 - \beta) z + \beta \theta_t`.
    It is created per parameter at the first step that parameter has a gradient, so
    param groups added to the wrapped optimizer later are covered too. Any other
    attribute is delegated to the wrapped optimizer.

    :param optimizer: The primal optimizer to wrap.
    :type optimizer: torch.optim.Optimizer
    :param mu: Smoothing multiplier; defaults to 2.0.
    :type mu: float
    :param beta: Prox-centre update rate; defaults to 0.5.
    :type beta: float
    :param primal_update_process_length: Number of primal steps per prox-centre update; defaults to 1.
    :type primal_update_process_length: int
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        mu: float = 2.0,
        beta: float = 0.5,
        primal_update_process_length: int = 1,
    ) -> None:
        if mu < 0:
            raise ValueError(
                f"The smoothing parameter`mu`must be non-negative, got {mu}."
            )
        self.optimizer = optimizer
        self.mu, self.beta = mu, beta
        self.primal_update_process_length = primal_update_process_length
        self.iter = 0
        self._z = {}

    def step(self) -> None:
        with torch.no_grad():
            # add smoothing term gradient to the gradient w.r.t. primal params, and update smoothing params before optimizer step
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is None:
                        continue
                    if param not in self._z:
                        self._z[param] = param.detach().clone()
                    smoothing_buffer = self._z[param]
                    param.grad.add_(param, alpha=self.mu).add_(
                        smoothing_buffer, alpha=-self.mu
                    )

                    # update only after n inner iterations
                    if self.iter + 1 == self.primal_update_process_length:
                        smoothing_buffer.add_(smoothing_buffer, alpha=-self.beta).add_(
                            param, alpha=self.beta
                        )

        self.optimizer.step()
        self.iter = (self.iter + 1) % self.primal_update_process_length

    def __getattr__(self, name):
        # Missing only after unpickling without __getstate__; guards against
        # infinite recursion through self.optimizer.
        if name == "optimizer":
            raise AttributeError(name)
        return getattr(self.optimizer, name)

    def __getstate__(self):
        # Optimizer.__getstate__ keeps only defaults/state/param_groups, which here
        # live on the wrapped optimizer.
        return self.__dict__.copy()

    def __setstate__(self, state):
        # Bypasses Optimizer.__setstate__, whose _patch_step_function would wrap
        # MoreauEnvelope.step class-wide and run the inner optimizer's hooks twice.
        self.__dict__.update(state)

    # TODO: do we need to save the smoothing buffer?

    # def state_dict(self) -> dict[str, Any]:

    #     dual_packed_state = {"smoothing_buffer": self.smoothing_buffer, "mu": self.mu, "beta": self.beta}
    #     dual_state_dict = {"state": dual_packed_state, "param_groups": self.param_groups}
    #     state_dict = {"primal": self.optimizer.state_dict(), "dual": dual_state_dict}
    #     return state_dict

    # def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    #     primal_state_dict = state_dict["primal"]
    #     self.primal_optimizer.load_state_dict(primal_state_dict)

    #     dual_state_dict = state_dict["dual"]
    #     self.penalty = dual_state_dict["state"]["penalty"]
    #     self.dual_range = dual_state_dict["state"]["dual_range"]
    #     self.smoothing_buffer = state_dict["state"]["smoothing_buffer"]
    #     params = dual_state_dict["param_groups"]
    #     self.param_groups = []
    #     for param in params:
    #         self.param_groups.append(param)
