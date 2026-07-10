#!/usr/bin/env python3
"""sorte a dataframe by high price in desecending"""


def high(df):
     #sort in desending order High
     return df.sort_values(by="High", ascending=False)
