#!/usr/bin/env python3
"""this module to calculate the likelihood"""
import numpy as np

def likelihood(x,n,P):
    """Arg:
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various 
    hypothetical probabilities of developing severe side effects
    
"""
    if type(n) is not int or n <= 0: 
         raise ValueError("n must be a positive integer")
    
    if type(x) is not int or x < 0:
        raise  ValueError("x must be an integer that is greater than "
        "or equal to 0")
    
    if not isinstance (P,np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")
    
    if not np.all((P >= 0)& (P <= 1)):
        raise ValueError("All values in P must be in the range [0,1]")


    return likelihood(x,n,np.any(P))
