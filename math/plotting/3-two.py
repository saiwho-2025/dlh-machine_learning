#!/usr/bin/env python3
"""plot 2 line graphs sharing the same x-axis."""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """plot 2 graphs x-y1, x-y2."""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    
    # Mathematical definitions for the decay curves
    # (Using standard half-life constants for C-14 and Ra-226)
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    #Initialize the figure size
    plt.figure(figsize=(6.4, 4.8))

    # Plot y1 line exactly as a dashed red line
    plt.plot(x, y1, "r--", label='C-14')

    # Plot y2 line exatly as a solid green line
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Set exact labels, title, and scaling
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")

    # Set the precise x-axis limits matching the test suite format
    plt.xlim([0, 20000])
    plt.ylim([0,1])

    # Place the legend in the upper right hand corner
    plt.legend(loc='upper right')

    # Render the graph
    plt.show()

    if __name__ == "__main__":
        two()
