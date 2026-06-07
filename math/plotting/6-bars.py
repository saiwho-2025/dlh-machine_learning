#!/usr/bin/env python3
"""plot a bar graph"""
import matplotlib.pyplot as plt
import numpy as np

def bars():
    # Define data
    fruit = np.array([
        [10, 12, 15], # apples
        [15, 10, 20], # bananas
        [5, 8, 10],   # oranges
        [20, 15, 5]   # peaches
    ])

    people = ['Farrah', 'Fred', 'Felicia']
    fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    plt.figure(figsize=(8, 6))
    bottom = np.zeros(len(people))

    for i in range(len(fruit)):
        plt.bar(people, fruit[i], bottom=bottom, color=colors[i], label=fruit_names[i], width=0.5)
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    
    # Depending on your environment, you might need plt.show() or return the figure
    plt.show()

# If running as a script directly
if __name__ == "__main__":
    bars()