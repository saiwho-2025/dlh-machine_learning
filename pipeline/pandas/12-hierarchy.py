#!/usr/bin/env python3
"""layers and order the work based on previous work"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """layers and order the work based on previous work"""
    df1 = index(df1)
    df2 = index(df2)

    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    return df.swaplevel(0, 1).sort_index()
