#!/usr/bin/env python3
"""Slice selected columns from a pandas DataFrame."""


def slice(df):
    """take a pd.DataFrame and extract columns, select the 60th row and return the sliced Frame"""
    return df[["High", "Low", "Close", "Volume_(BTC)"]].row(60).iloc[::60]
