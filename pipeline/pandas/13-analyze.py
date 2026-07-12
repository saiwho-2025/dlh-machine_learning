#!/usr/bin/env python3
"""computes descriptive statistics for  all columns excepte the Timestamp"""


def analyze(df):
    """calculate except 1 column"""
    return df.drop(columns=["Timestamp"]).describe()
