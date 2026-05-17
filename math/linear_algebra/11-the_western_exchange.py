#!/usr/bin/env python3
"""exchange the rows of a matrix"""

import numpy as np


def np_transpose(matrix):
    """Transpose a matrix."""
    return [list(row) for row in zip(*matrix)]
