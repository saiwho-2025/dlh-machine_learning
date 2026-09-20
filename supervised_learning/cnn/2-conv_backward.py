#!/usr/bin/env python3
"""Module containing the conv_backward function."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph = 0
        pw = 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    h_new, w_new = dZ.shape[1], dZ.shape[2]

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                h_start = h * sh
                h_end = h_start + kh
                w_start = w * sw
                w_end = w_start + kw

                a_slice = a_prev_pad[
                    h_start:h_end,
                    w_start:w_end,
                    :
                ]

                for c in range(c_new):
                    da = dZ[i, h, w, c]

                    dW[:, :, :, c] += a_slice * da
                    db[:, :, :, c] += da

                    da_prev_pad[
                        h_start:h_end,
                        w_start:w_end,
                        :
                    ] += W[:, :, :, c] * da

        dA_prev_pad[i] = da_prev_pad

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
