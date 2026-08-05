#!/usr/bin/env python3
"""Select a Gaussian mixture model using BIC."""

# Import NumPy to handle numerical arrays.
import numpy as np

# Load the required EM function.
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(
    X,
    kmin=1,
    kmax=None,
    iterations=1000,
    tol=1e-5,
    verbose=False
):
    """
    Select the best GMM cluster count using BIC.

    Args:
        X: NumPy array with shape (n, d).
        kmin: Smallest cluster count.
        kmax: Largest cluster count.
        iterations: Maximum EM update count.
        tol: Nonnegative EM tolerance.
        verbose: Boolean controlling EM output.

    Returns:
        best_k, best_result, l, and b.
    """
    # Validate X.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    # Read the data dimensions.
    n, d = X.shape

    # Reject an empty data set.
    if n == 0 or d == 0:
        return None, None, None, None

    # Validate kmin.
    if (
        not isinstance(kmin, int)
        or isinstance(kmin, bool)
        or kmin <= 0
    ):
        return None, None, None, None

    # Use the data-point count when kmax is omitted.
    if kmax is None:
        kmax = n

    # Validate kmax.
    if (
        not isinstance(kmax, int)
        or isinstance(kmax, bool)
        or kmax <= 0
    ):
        return None, None, None, None

    # Require at least two cluster counts.
    if kmax <= kmin:
        return None, None, None, None

    # Validate iterations.
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations <= 0
    ):
        return None, None, None, None

    # Validate tol.
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    # Validate verbose.
    if not isinstance(verbose, bool):
        return None, None, None, None

    # Calculate the output size.
    size = kmax - kmin + 1

    # Create the likelihood array.
    l = np.zeros(size)

    # Create the BIC array.
    b = np.zeros(size)

    # Create storage holding each model.
    results = []

    # Test each cluster count.
    for index, clusters in enumerate(range(kmin, kmax + 1)):
        # Run expectation maximization.
        pi, m, S, g, likelihood = expectation_maximization(
            X,
            clusters,
            iterations,
            tol,
            verbose
        )

        # Check the EM result.
        if (
            pi is None
            or m is None
            or S is None
            or g is None
            or likelihood is None
        ):
            return None, None, None, None

        # Save the likelihood.
        l[index] = likelihood

        # Calculate the parameter count.
        parameters = (
            clusters - 1
            + clusters * d
            + clusters * d * (d + 1) // 2
        )

        # Calculate the BIC value.
        b[index] = (
            parameters * np.log(n)
            - 2 * likelihood
        )

        # Save the current model.
        results.append((pi, m, S))

    # Locate the smallest BIC value.
    best_index = np.argmin(b)

    # Calculate the best cluster count.
    best_k = kmin + best_index

    # Select the best model.
    best_result = results[best_index]

    # Return the selection and measurements.
    return best_k, best_result, l, b
