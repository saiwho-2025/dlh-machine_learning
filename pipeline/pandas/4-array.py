#!/usr/bin/env python3
"""take a pd.DataFrame as input, select function, convert into ndarray"""

import pandas as pd


def array(df):
    """Return the last 10 rows of High and Close as a NumPy array."""
    return df[["High", "Close"]].tail(10).to_numpy()
