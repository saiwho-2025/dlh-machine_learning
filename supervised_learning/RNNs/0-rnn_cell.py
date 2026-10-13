#!/usr/bin/env python3
"""Simple RNN cell implementation."""

import numpy as np


class RNNCell:
    """Represent a cell of a simple recurrent neural network."""

    def __init__(self, i, h, o):
        """Initialize the weights and biases of the RNN cell."""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        # Concatenate the previous hidden state and current input.
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Calculate the next hidden state using tanh activation.
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Calculate the output using softmax activation.
        z = np.matmul(h_next, self.Wy) + self.by

        # Apply numerically stable softmax.
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y
