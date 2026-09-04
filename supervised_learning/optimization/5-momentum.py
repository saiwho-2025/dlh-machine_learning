#!/usr/bin/env python3
"""using gradient descent to update a variable in momentum optimization algorithm"""

import numpy as np

def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using gradient descent with momentum."""
    v = beta *v + (1 - beta1) * grad
    var = var - alpha *v

    return var, v
