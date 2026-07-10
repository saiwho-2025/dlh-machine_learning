#!/usr/bin/env python3
"""sorte a dataframe by high price in desecending"""


def high(df):
     """Return the DataFrame sorted by High in descending order."""
     return df.sort_values(by="High", ascending=False)
