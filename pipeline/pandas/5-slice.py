#!/usr/bin/env python3
"""Slice selected columns from a pandas DataFrame."""


def slice(df):
    """take a pd.DataFrame, extract columns, select the 60th row and return it"""
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
