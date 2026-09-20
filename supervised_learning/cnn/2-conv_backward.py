#!/usr/bin/env python3
"""Convolutional layer backward propagation."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Perform back propagation over a convolutional layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    _, h_new, w_new, c_new = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Calculate the padding.
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph = 0
        pw = 0

    # Pad the previous layer.
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    # Initialize the gradients.
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # Calculate the gradients.
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                h_start = h * sh
                h_end = h_start + kh
                w_start = w * sw
                w_end = w_start + kw

                a_slice = A_prev_pad[
                    i,
                    h_start:h_end,
                    w_start:w_end,
                    :
                ]

                for c in range(c_new):
                    dz = dZ[i, h, w, c]

                    dA_prev_pad[
                        i,
                        h_start:h_end,
                        w_start:w_end,
                        :
                    ] += W[:, :, :, c] * dz

                    dW[:, :, :, c] += a_slice * dz
                    db[:, :, :, c] += dz

    # Remove the padding.
    if padding == "same":
        dA_prev = dA_prev_pad[
            :,
            ph:ph + h_prev,
            pw:pw + w_prev,
            :
        ]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
