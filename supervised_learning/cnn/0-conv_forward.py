    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1

    A_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )
