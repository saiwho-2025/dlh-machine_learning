#!/usr/bin/env python3
"""Pooling layer backward propagation."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform back propagation over a pooling layer."""
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize the gradient with respect to the previous layer.
    dA_prev = np.zeros_like(A_prev)

    # Calculate the gradient for each pooling window.
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                h_start = h * sh
                h_end = h_start + kh
                w_start = w * sw
                w_end = w_start + kw

                for ch in range(c):
                    if mode == 'max':
                        # Select the values that produced the maximum.
                        a_slice = A_prev[
                            i,
                            h_start:h_end,
                            w_start:w_end,
                            ch
                        ]
                        mask = a_slice == np.max(a_slice)

                        dA_prev[
                            i,
                            h_start:h_end,
                            w_start:w_end,
                            ch
                        ] += mask * dA[i, h, w, ch]

                    elif mode == 'avg':
                        # Distribute the gradient evenly over the window.
                        da = dA[i, h, w, ch] / (kh * kw)

                        dA_prev[
                            i,
                            h_start:h_end,
                            w_start:w_end,
                            ch
                        ] += np.ones((kh, kw)) * da

    return dA_prev
