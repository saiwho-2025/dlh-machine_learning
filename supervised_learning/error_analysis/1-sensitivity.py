#!/usr/bin/env python3
"""Calculates the sensitivity for each class."""

import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class.

    Args:
        confusion: A confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels.

    Returns:
        A numpy.ndarray of shape (classes,) containing the sensitivity
        of each class.
    """
    # Get the true positives for each class.
    true_positives = np.diag(confusion)

    # Get the total number of actual examples for each class.
    actual = np.sum(confusion, axis=1)

    # Calculate sensitivity for each class.
    return true_positives / actual
