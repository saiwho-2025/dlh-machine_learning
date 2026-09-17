#!/usr/bin/env python3
"""Calculate the F1 score for each class."""

import numpy as np


def f1_score(confusion):
    """Return the F1 score for each class."""
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    sen = sensitivity(confusion)
    pre = precision(confusion)

    return 2 * sen * pre / (sen + pre)
