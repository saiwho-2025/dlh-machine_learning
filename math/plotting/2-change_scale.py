#!/usr/bin/env python3
"""This module plots the exponential decay of Carbon-14."""

import matplotlib.pyplot as plt
import numpy as np


def change_scale():
    # 1. Setup the data
    # Carbon-14 half-life is 5730 years
    half_life = 5730
    x = np.arange(0, 28651, half_life)
    
    # Mathematical calculation of decay
    r = np.log(0.5)
    y = np.exp((r / half_life) * x)

    # 2. Initialize the plot
    plt.figure(figsize=(6.4, 4.8))
    
    # 3. Plot the data (Crucial missing step!)
    plt.plot(x, y, marker='o', linestyle='-', color='b') 
    
    # 4. Customize axes and labels
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")
    plt.yscale("log")
    plt.xlim(0, 28651)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)  # Added for log-scale readability

    # 5. Display the plot
    plt.show()  # Added parentheses to execute the function

if __name__ == "__main__":
    change_scale()
