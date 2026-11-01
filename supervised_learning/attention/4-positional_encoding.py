import numpy as np


def positional_encoding(max_seq_len, dm):
    """Return a sinusoidal encoding of shape (max_seq_len, dm)."""
    positions = np.arange(max_seq_len)[:, None]
    dimensions = np.arange(dm)
    exponents = 2 * (dimensions // 2) / dm
    encoding = positions / (10000.0 ** exponents)

    encoding[:, 0::2] = np.sin(encoding[:, 0::2])
    encoding[:, 1::2] = np.cos(encoding[:, 1::2])
    return encoding
