import matplotlib.pyplot as plt
import numpy as np

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# Highs and Lows for each day
temps = [
    [25, 28, 26, 24, 27, 30, 29], # Highs
    [15, 17, 16, 14, 18, 20, 19]  # Lows
]
colors = ['#FF5733', '#33C1FF']
labels = ['High', 'Low']

# Set the width of the bars
bar_width = 0.35
# Create an array for the x-axis positions
x = np.arange(len(days))

# --- YOUR TASK: Use a for loop to plot the two sets of bars ---
# Hint: You will need to offset 'x' by the bar_width for each iteration
# to keep them side-by-side rather than stacked.

for i in range(2):
    # Adjust position for each set of bars
    offset = x + (i * bar_width)
    plt.bar(offset, temps[i], width=bar_width, color=colors[i], label=labels[i])

plt.xticks(x + bar_width/2, days)
plt.legend()
plt.show()