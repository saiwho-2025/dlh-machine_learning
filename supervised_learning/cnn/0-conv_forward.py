#!/usr/bin/env python3
"""Convolutional layer forward propagation."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Perform forward propagation over a convolutional layer."""
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        h_new = int(np.ceil(h_prev / sh))
        w_new = int(np.ceil(w_prev / sw))

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

        h_new = (h_prev - kh) // sh + 1
        w_new = (w_prev - kw) // sw + 1
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # Pad the input with zeros.
    A_pad = np.pad(
        A_prev,
        (
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0)
        ),
        mode="constant"
    )

    # Initialize the convolution output.
    Z = np.zeros((m, h_new, w_new, c_new))

    # Perform the convolution.
    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extract the current slice from the padded input.
            window = A_pad[:, h_start:h_end, w_start:w_end, :]

            # Apply all filters to the current slice.
            Z[:, i, j, :] = np.tensordot(
                window,
                W,
                axes=([1, 2, 3], [0, 1, 2])
            ) + b.reshape(1, c_new)

    # Apply the activation function.
    return activation(Z)
