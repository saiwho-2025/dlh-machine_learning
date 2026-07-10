#!/usr/bin/env python3
"""Sort a DataFrame by the High column in descending order."""


def high(df):
     """Return the DataFrame sorted by High in descending order."""
     return df.sort_values(by="High", ascending=False)
