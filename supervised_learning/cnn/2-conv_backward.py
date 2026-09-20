#!/usr/bin/env python3
"""Module containing the conv_backward function."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    h_new, w_new = dZ.shape[1], dZ.shape[2]

    if padding == "same":
        pad_h = max((h_new - 1) * sh + kh - h_prev, 0)
        pad_w = max((w_new - 1) * sw + kw - w_prev, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

    elif padding == "valid":
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0

    else:
        raise ValueError("padding must be 'same' or 'valid'")

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

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                h_start = h * sh
                w_start = w * sw

                h_end = h_start + kh
                w_end = w_start + kw

                # Clip the window to the padded input
                h_start_clip = max(h_start, 0)
                h_end_clip = min(h_end, A_prev_pad.shape[1])
                w_start_clip = max(w_start, 0)
                w_end_clip = min(w_end, A_prev_pad.shape[2])

                # Corresponding kernel region
                kh_start = h_start_clip - h_start
                kh_end = kh_start + (h_end_clip - h_start_clip)
                kw_start = w_start_clip - w_start
                kw_end = kw_start + (w_end_clip - w_start_clip)

                a_slice = A_prev_pad[
                    i,
                    h_start_clip:h_end_clip,
                    w_start_clip:w_end_clip,
                    :
                ]

                for c in range(c_new):
                    dz = dZ[i, h, w, c]

                    dW[
                        kh_start:kh_end,
                        kw_start:kw_end,
                        :,
                        c
                    ] += a_slice * dz

                    db[:, :, :, c] += dz

                    dA_prev_pad[
                        i,
                        h_start_clip:h_end_clip,
                        w_start_clip:w_end_clip,
                        :
                    ] += (
                        W[
                            kh_start:kh_end,
                            kw_start:kw_end,
                            :,
                            c
                        ] * dz
                    )

    dA_prev = dA_prev_pad[
        :,
        pad_top:pad_top + h_prev,
        pad_left:pad_left + w_prev,
        :
    ]

    return dA_prev, dW, db
