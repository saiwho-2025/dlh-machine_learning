#!/usr/bin/env python3
"""remove NaN values in Close"""


def prune(df):
    """remove in Close the NaN values"""
    return df.dropna(subset=["Close"])
