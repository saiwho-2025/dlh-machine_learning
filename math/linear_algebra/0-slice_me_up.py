#!/usr/bin/env python3
import numpy as np


# Extract specific row and column slices from the matrix
C = np.arange(50).reshape(5, 10); a1, a2, a3 = C[0:1], C[0:4], C[:, 3:7]
print(a1, a2, a3, sep="\n")
