#!/usr/bin/env python3
"""Convolutional layer forward propagation."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Perform forward propagation using symmetric zero padding."""
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        # Round up the padding per side to keep it symmetric.
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # Count complete filter windows after padding and striding.
    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1

    # Pad only the spatial axes; preserve the batch and channel axes.
    A_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        h_start = i * sh
        h_end = h_start + kh

        for j in range(w_new):
            w_start = j * sw
            w_end = w_start + kw
            # Extract the same spatial window from every image in the batch.
            a_slice = A_pad[:, h_start:h_end, w_start:w_end, :]

            for k in range(c_new):
                # Sum over height, width, and input channels for each image.
                Z[:, i, j, k] = (
                    np.sum(a_slice * W[:, :, :, k], axis=(1, 2, 3))
                    + b[0, 0, 0, k]
                )

    # Apply the activation after each weighted sum includes its bias.
    return activation(Z)
