#!/usr/bin/env python3
"""Calculates the precision for each class."""

import numpy as np

def precision(confusion):
    """Calculates the precision for each class.
    Args:
    confusion: A confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column
        indices represent the predicted labels.

    Returns:
    A numpy.ndarray of shape (classes,) containing the precision
    of each class.
"""
    # Get the true positives for each class.
    true_positives = np.diag(confusion)

    # Get the total number of predictions for each class.
    predicted = np.sum(confusion, axis=0)

    # Calculate precision for each class.
    return true_positives / predicted

