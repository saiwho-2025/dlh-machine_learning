#!/usr/bin/env python3
"""plot a scatter of sampled elevations on a mountain"""
import numpy as np
import matplotlib.pyplot as plt

def gradient():
    """plot the colorbars"""
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    
    # initiate the graph
    plt.figure(figsize=(6.4, 4.8))

    # scatter setting
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, c = z)

    # labels and title setting
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")
    plt.title("Mountain Elevation")

    # show it
    plt.show()

