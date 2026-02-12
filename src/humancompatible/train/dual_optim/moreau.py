import torch
from torch.optim import Optimizer
from torch.optim.optimizer import _use_grad_for_differentiable


class MoreauEnvelope(Optimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        mu: float = 2.,
        beta: float = 0.5,
    ) -> None:
        """
        A wrapper over a PyTorch`Optimizer` that allows for quick calculation of the Moreau envelope gradient.

        :param optimizer: Optimizer.
        :type torch.optim.Optimizer: int
        :param mu: Smoothing multiplier; defaults to 0.
        :type mu: float
        :param beta: Smoothing multiplier update rate; defaults to 0.5.
        :type beta: float
        """

        self.optimizer = optimizer
        self.mu, self.beta = mu, beta

        if mu <= 0:
            raise ValueError(f"The smoothing parameter`mu`should be positive, got {mu}.")
        else:
            self.smoothing_buffer = []
            for param_group in optimizer.param_groups:
                self.smoothing_buffer.append({'params': []})
                for _, param in enumerate(param_group['params']):
                    self.smoothing_buffer[-1]['params'].append(param.clone().detach())

    def step(self) -> None:
        with torch.no_grad():
            # add smoothing term gradient to the gradient w.r.t. primal params, and update smoothing params before optimizer step
            for param_group, smoothing_buffer_group in zip(self.optimizer.param_groups, self.smoothing_buffer):
                for param, smoothing_buffer in zip(param_group["params"], smoothing_buffer_group['params']):
                    param.grad.add_(param, alpha=self.mu).add_(smoothing_buffer, alpha=-self.mu)
                    smoothing_buffer.add_(smoothing_buffer, alpha=-self.beta).add_(param, alpha=self.beta)
        
        self.optimizer.step()


    def __getattr__(self, name):
        # Delegate to the wrapped object
        try:
            attr = getattr(self.optimizer, name)
        except AttributeError:
            raise AttributeError(f"'A' object has no attribute '{name}'")
        else:
            # If it's a method, bind it to self.optimizer
            if callable(attr):
                def method(*args, **kwargs):
                    return attr(self.optimizer, *args, **kwargs)
                return method
            else:
                return attr

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