#!/usr/bin/env python3
"""Calculate specificity for each class in a confusion matrix."""

import numpy as np


def specificity(confusion):
    """Return the specificity for each class."""
    total = np.sum(confusion)
    result = np.empty(confusion.shape[0], dtype=float)

    for i in range(confusion.shape[0]):
        true_negative = (
            total
            - np.sum(confusion[i, :])
            - np.sum(confusion[:, i])
            + confusion[i, i]
        )
        false_positive = np.sum(confusion[:, i]) - confusion[i, i]

        result[i] = true_negative / (true_negative + false_positive)

    return result
