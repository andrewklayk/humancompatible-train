import torch

"""
Module with the implementation of penalty/barrier functions.

A more detailed description of the properties of the barrier functions is listed here: 
https://www.researchgate.net/publication/2775406_PenaltyBarrier_Multiplier_Methods_for_Convex_Programming_Problems

"""

def exponential_penalty(t):

    return torch.exp(t) - 1.0

def modified_log_barrier(t):

    return -torch.log(1-t)

def augmented_lagrangian(t):
    """
    Vectorized version of augmented_lagrangian
    """
    return torch.where(
        t >= -1,
        t + 0.5 * torch.square(t),
        -0.5 * torch.ones_like(t)
    )


def quad_log(t):
    """
    Vectorized version of quadratic_logarithmic_penalty
    """

    out = torch.empty_like(t)

    mask = t >= -0.5
    out[mask] = t[mask] + 0.5 * torch.pow(t[mask], 2)
    out[~mask] = -0.25 * torch.log(-2 * t[~mask]) - 3/8

    return out


def quad_recipr(t):
    """
    Vectorized version of quadratic_reciprocal_penalty
    """

    out = torch.empty_like(t)

    mask = t >= -1/3
    out[mask] = t[mask] + 0.5 * torch.pow(t[mask], 2)
    out[~mask] = (32/27) * (1 / (1 - t[~mask])) - 7/6

    return out  

def exponential_penalty_derivative(t):

    return torch.exp(t) 

def modified_log_barrier_derivative(t):

    return 1 / (1-t)

def aug_lagr_der(t):

    return torch.where(
        t >= -1,
        1 + t,
        torch.zeros_like(t)
    )


def quad_log_der(t):

    out = torch.empty_like(t)

    mask = t >= -0.5
    out[mask] = 1 + t[mask]
    out[~mask] = -1 / (4 * t[~mask])

    return out

def quad_recipr_der(t):

    out = torch.empty_like(t)

    mask = t >= -1/3
    out[mask] = 1 + t[mask]
    out[~mask] = (32/27) * (1 / torch.square(1 - t[~mask]))

    return out