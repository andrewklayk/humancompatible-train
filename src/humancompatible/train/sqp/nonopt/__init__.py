"""
PyTorch port of NonOpt (https://frankecurtis.github.io/NonOpt/), a solver for
unconstrained, locally Lipschitz (possibly nonconvex, nonsmooth) minimization
by Frank E. Curtis and collaborators (Curtis & Zebiane, arXiv:2503.22826).
"""

from .direction import CuttingPlane, GradientCombination, GradientDirection
from .inverse_hessian import DenseInverseHessian, LimitedMemoryInverseHessian
from .line_search import backtracking, weak_wolfe
from .optimizer import NonOpt
from .point_set import Point, PointSet
from .qp import project_onto_simplex, solve_simplex_qp
