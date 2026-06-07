#!/usr/bin/env python3
"""stack of bars in name of people and fruits"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ plots stack of bars, with semantic color choice"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    # initialize the graph
    plt.figure(figsize=(6.4, 4.8))

    #  setting the bar tick labels
    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    width = 0.50

    # logical process of bar plotting
    bottom = np.zeros(3)
    for i in range(4):
        plt.bar(people, fruit[i], width=width, color=colors[i],
                label=labels[i], bottom=bottom)
        bottom += fruit[i]

    # setting the graph labels and titles
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.title('Number of Fruit per Person')

    # legend is required
    plt.legend()

    # show the plot
    plt.show()
