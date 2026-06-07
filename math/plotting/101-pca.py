#!/usr/bin/env python3
"""visualize the Iris Flower data set"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with plasma colormap
scatter = ax.scatter(
    pca_data[:, 0], 
    pca_data[:, 1], 
    pca_data[:, 2], 
    c=labels, 
    cmap='plasma'
)

# Labeling as requested
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.title('PCA of Iris Dataset')

# Add a colorbar to indicate the species (labels 0, 1, 2)
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'])

plt.show()