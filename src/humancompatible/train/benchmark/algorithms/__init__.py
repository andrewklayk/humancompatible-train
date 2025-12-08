from .ghost import StochasticGhost
from .ssl_alm import SSLALM
from .switching_subgradient import SSG
from .sgd import SGD
# from .torch.ssl_alm import SSLALM
# from .torch.ssw import SSG

__all__ = ["SSLALM", "StochasticGhost", "SSG", "SGD"]
