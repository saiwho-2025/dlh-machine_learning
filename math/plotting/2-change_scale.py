#!/usr/bin/env python3
"""This module plots the exponential decay of C-14"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Plots the exponential decay of C-14 with a logarithmic scale"""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    # Plot the line exactly as a solid blue line
    plt.plot(x, y)

    # Set exact labels, title, and scaling
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")
    plt.yscale("log")

    # Set the precise x-axis limits matching the test suite format
    plt.xlim([0, 28650])

    plt.show()
