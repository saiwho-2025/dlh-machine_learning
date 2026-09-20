#!/usr/bin/env python3
"""Module containing the pool_forward function."""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            a_slice = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(a_slice, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(a_slice, axis=(1, 2))

    return A
