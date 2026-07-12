#!/usr/bin/env python3
"""computes descriptive statistics for  all columns excepte the Timestamp"""


def analyze(df):
    return df.drop(columns=["Timestamp"]).describe()
