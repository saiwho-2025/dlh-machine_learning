#!/usr/bin/env python3
"""take a pd.DataFrame and set column as index"""


def index(df):
    """set the Timestamp column as index"""
    return df.set_index("Timestamp")
