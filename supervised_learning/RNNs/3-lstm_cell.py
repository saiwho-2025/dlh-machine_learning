#!/usr/bin/env python3
"""Defines an LSTM cell"""

import numpy as np


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """Initializes the LSTM cell.

        Args:
            i: Dimensionality of the input data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the output.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step.

        Args:
            h_prev: Previous hidden state of shape (m, h).
            c_prev: Previous cell state of shape (m, h).
            x_t: Input data of shape (m, i).

        Returns:
            h_next: Next hidden state.
            c_next: Next cell state.
            y: Output of the cell.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f = 1 / (1 + np.exp(-(concat @ self.Wf + self.bf)))

        # Update gate
        u = 1 / (1 + np.exp(-(concat @ self.Wu + self.bu)))

        # Intermediate cell state
        c = np.tanh(concat @ self.Wc + self.bc)

        # Output gate
        o = 1 / (1 + np.exp(-(concat @ self.Wo + self.bo)))

        # Next cell state
        c_next = f * c_prev + u * c

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Softmax output
        z = h_next @ self.Wy + self.by
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, c_next, y
