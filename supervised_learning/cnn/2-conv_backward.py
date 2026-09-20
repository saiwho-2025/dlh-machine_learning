#!/usr/bin/env python3
"""Module containing the conv_backward function."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    h_new, w_new = dZ.shape[1], dZ.shape[2]

    # Calculate padding
    if padding == "same":
        pad_h = max((h_new - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_new - 1) * sw + kw - w_prev, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    else:
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0

    # Pad A_prev
    A_prev_pad = np.pad(
        A_prev,
        (
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0)
        ),
        mode="constant"
    )

    # Initialize gradients
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # Back propagation
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

                    dW[:, :, :, c] += a_slice * dz
                    db[:, :, :, c] += dz

                    dA_prev_pad[
                        i,
                        h_start:h_end,
                        w_start:w_end,
                        :
                    ] += W[:, :, :, c] * dz

    # Remove padding
    dA_prev = dA_prev_pad[
        :,
        pad_top:pad_top + h_prev,
        pad_left:pad_left + w_prev,
        :
    ]

    return dA_prev, dW, db
