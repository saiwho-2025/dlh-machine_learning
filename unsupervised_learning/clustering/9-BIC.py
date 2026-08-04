#!/usr/bin/env python3
"""Select a GMM cluster count using BIC."""

# Import NumPy to handle numerical arrays.
import numpy as np

# Load the required EM function.
expectation_maximization = __import__(
    '8-EM'
).expectation_maximization


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
        Best cluster count, model, likelihoods, and BIC values.
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

    # Use the point count when kmax is omitted.
    if kmax is None:
        kmax = n

    # Validate kmax.
    if (
        not isinstance(kmax, int)
        or isinstance(kmax, bool)
        or kmax <= 0
    ):
        return None, None, None, None

    # Validate the cluster-count interval.
    if kmax < kmin:
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

    # Calculate the number of tested cluster counts.
    count = kmax - kmin + 1

    # Create the likelihood output array.
    l = np.empty(count)

    # Create the BIC output array.
    b = np.empty(count)

    # Create storage holding each fitted model.
    results = []

    # Test each cluster count.
    for index, cluster_count in enumerate(
        range(kmin, kmax + 1)
    ):
        # Run expectation maximization.
        pi, m, S, g, log_likelihood = expectation_maximization(
            X,
            cluster_count,
            iterations,
            tol,
            verbose
        )

        # Check the EM output.
        if (
            pi is None
            or m is None
            or S is None
            or g is None
            or log_likelihood is None
        ):
            return None, None, None, None

        # Save the current log likelihood.
        l[index] = log_likelihood

        # Count the independent model parameters.
        parameters = (
            cluster_count - 1
            + cluster_count * d
            + cluster_count * d * (d + 1) // 2
        )

        # Calculate and save the current BIC value.
        b[index] = (
            parameters * np.log(n)
            - 2 * log_likelihood
        )

        # Save the current priors, means, and covariances.
        results.append((pi, m, S))

    # Locate the smallest BIC value.
    best_index = int(np.argmin(b))

    # Convert the array index into a cluster count.
    best_k = kmin + best_index

    # Select the corresponding model.
    best_result = results[best_index]

    # Return the best model and all measurements.
    return best_k, best_result, l, b
