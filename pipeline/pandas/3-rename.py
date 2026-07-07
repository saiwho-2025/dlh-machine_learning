#!/usr/bin/env python3
"""take a pd.DataFrame as input and display it by specifications"""

import pandas as pd


# create DataFrame
def rename(df):
    """Rename Timestamp, convert it to datetime, and select two columns."""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
