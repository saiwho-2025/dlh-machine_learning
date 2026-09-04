import numpy as np


#!/usr/bin/env python3
"""this module calculates the weighted moving average of a data set"""


def moving_average(data,beta):
    """this module trains by the weighted moving average
    args:
        data: list of data to calculate the moving average 
        beta: the weight used for the moving average"""
    averages = []
    v =  0

    for t, value in enumerate(data,1):
        v = beta * v + (1- beta) * value
        v_corrected = v / (1- beta ** t)
        averages.append(v_corrected)

    return averages
