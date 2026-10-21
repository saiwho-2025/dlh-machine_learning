#!/usr/bin/env python3
"""Performs forward propagation for a simple RNN"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: Instance of RNNCell.
        X: Input data of shape (t, m, i).
        h_0: Initial hidden state of shape (m, h).

    Returns:
        H: All hidden states of shape (t + 1, m, h).
        Y: All outputs of shape (t, m, o).
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y = []

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
