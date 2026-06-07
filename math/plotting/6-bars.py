#!/usr/bin/env python3
"""plot a stacked bar graph"""
import matplotlib.pyplot as plt
import numpy as np

# Sample data:
# Rows: apples, bananas, oranges, peaches
# Columns: Farrah, Fred, Felicia
fruit = np.array([
    [10, 12, 15], # apples
    [15, 10, 20], # bananas
    [5, 8, 10],   # oranges
    [20, 15, 5]   # peaches
])

# Define labels and colors
people = ['Farrah', 'Fred', 'Felicia']
fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Initialize the plot
plt.figure(figsize=(8, 6))

# Initialize an array of zeros to track the bottom position of the stack
bottom = np.zeros(len(people))

# Plot each row of the matrix as a stacked bar
for i in range(len(fruit)):
    plt.bar(people, fruit[i], bottom=bottom, color=colors[i], label=fruit_names[i], width=0.5)
    # Update the bottom position for the next fruit in the stack
    bottom += fruit[i]

# Add labels and formatting
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.yticks(np.arange(0, 81, 10)) # Range from 0 to 80 with ticks every 10
plt.legend()

# Save or display the plot
plt.savefig('fruit_plot.png')