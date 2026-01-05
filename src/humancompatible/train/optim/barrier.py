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
    This penalty/barrier is equal to the standard augmented lagrangian multiplier method
    """

    if t >= -1:
        return t + 0.5*torch.square(t) 

    else: 
        return t*0 - 0.5

def quadratic_logarithmic_penalty(t):
    
    if t >= -0.5:
        return t + 0.5*torch.square(t) 

    else: 
        return -0.25 * torch.log(-2 * t) - 3/8


def quadratic_reciprocal_penalty(t):

    if t >= -1/3:
        return t + 0.5*torch.square(t) 
    else: 
        return (32/27) * (1/(1-t)) - 7/6
