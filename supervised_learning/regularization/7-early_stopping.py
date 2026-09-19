#!/usr/bin/env python3
"""Early stopping module."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if gradient descent should stop early."""
    if cost > opt_cost - threshold:
        count += 1
    else:
        count = 0

    if count >= patience:
        return True, count

    return False, count
