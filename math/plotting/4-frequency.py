#!/usr/bin/env python3
"""Plot frenquency groupped into bins"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Frenquency groupped into bins, blue ones"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # initialize the graph
    plt.figure(figsize=(6.4, 4.8))

    # Plot the bins
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # show the result
    plt.show()
