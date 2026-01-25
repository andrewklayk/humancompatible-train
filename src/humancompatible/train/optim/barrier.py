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

def augmented_lagrangian_deprecated(t):
    """
    This penalty/barrier is equal to the standard augmented lagrangian multiplier method
    """

    if t >= -1:
        return t + 0.5*torch.square(t) 

    else: 
        return t*0 - 0.5

def quadratic_logarithmic_penalty_deprecated(t):
    
    if t >= -0.5:
        return t + 0.5*torch.square(t) 

    else: 
        return -0.25 * torch.log(-2 * t) - 3/8


def quadratic_reciprocal_penalty_deprecated(t):

    if t >= -1/3:
        return t + 0.5*torch.square(t) 
    else: 
        return (32/27) * (1/(1-t)) - 7/6

# -------------------------------- vectorized versions

def augmented_lagrangian(t):
    """
    Vectorized version of augmented_lagrangian
    """
    return torch.where(
        t >= -1,
        t + 0.5 * torch.square(t),
        -0.5 * torch.ones_like(t)
    )


def quadratic_logarithmic_penalty(t):
    """
    Vectorized version of quadratic_logarithmic_penalty
    """

    return torch.where(
        t >= -0.5,
        t + 0.5 * torch.square(t),
        -0.25 * torch.log(-2 * t) - 3/8
    )


def quadratic_reciprocal_penalty(t):
    """
    Vectorized version of quadratic_reciprocal_penalty
    """
    return torch.where(
        t >= -1/3,
        t + 0.5 * torch.square(t),
        (32/27) * (1 / (1 - t)) - 7/6
    )

# ---------------------------------------------

def exponential_penalty_derivative(t):

    return torch.exp(t) 

def modified_log_barrier_derivative(t):

    return 1 / (1-t)

def augmented_lagrangian_derivative_deprecated(t):
    """
    This penalty/barrier is equal to the standard augmented lagrangian multiplier method
    """

    if t >= -1:
        return 1 + t

    else: 
        return t*0

def quadratic_logarithmic_penalty_derivative_deprecated(t):

    if t >= -0.5:
        return 1 + t

    else: 
        return -1/(4*t)
    

def quadratic_reciprocal_penalty_derivative_deprecated(t):

    if t >= -1/3:
        return 1 + t
    else: 
        return (32/27) * (1/torch.square(1-t))
    


# ------------------------------------ vectorized versions --------------------------



def augmented_lagrangian_derivative(t):
    return torch.where(
        t >= -1,
        1 + t,
        torch.zeros_like(t)
    )


def quadratic_logarithmic_penalty_derivative(t):

    return torch.where(
        t >= -0.5,
        1 + t,
        -1 / (4 * t)
    )


def quadratic_reciprocal_penalty_derivative(t):
    return torch.where(
        t >= -1/3,
        1 + t,
        (32/27) * (1 / torch.square(1 - t))
    )
