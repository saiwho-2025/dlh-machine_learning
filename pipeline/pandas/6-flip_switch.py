#!/usr/bin/env python3
"""reverse chronological order sort the dataframe """


def flip_switch(df):
    """sort the data in reverse chronoglical order,
    transposes the sorted dataframe
    return the transformed pd.DataFrame"""
    return df.iloc[::-1].T
