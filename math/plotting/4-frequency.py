#!/usr/bin/env python3
"""this module plot a histogram of student grades for a project"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    # plot student grades for a project

    np.random.seed(0)
    student_grades = np.random.normal(68, 15, 50)

    # Initialize the figure size
    plt.figure(figsize=(6.4, 4.8))

    # Define bins every 10 units from 0 to 100
    bins = np.arange(0, 101, 10)

    # Plot histogram with black outlines around the bars
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Label the axes and give the plot a title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Align the x-axis ticks with the 10-unit bins and set precise limits
    plt.xticks(bins)
    plt.xlim(0, 100)

    # Display the final graph
    plt.show()
