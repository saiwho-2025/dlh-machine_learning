#!/usr/bin/env python3
"""Run expectation maximization on a Gaussian mixture model."""

# Import NumPy to validate the data set.
import numpy as np

# Load the required initialization function.
initialize = __import__('4-initialize').initialize

# Load the required expectation function.
expectation = __import__('6-expectation').expectation

# Load the required maximization function.
maximization = __import__('7-maximization').maximization


def expectation_maximization(
    X,
    k,
    iterations=1000,
    tol=1e-5,
    verbose=False
):
    """
    Run the EM algorithm on a Gaussian mixture model.

    Args:
        X: NumPy array with shape (n, d).
        k: Positive cluster count.
        iterations: Maximum update count.
        tol: Nonnegative convergence tolerance.
        verbose: Boolean controlling likelihood output.

    Returns:
        pi, m, S, g, and l, or five None values on failure.
    """
    # Validate X.
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    # Read the data dimensions.
    n, d = X.shape

    # Reject an empty data set.
    if n == 0 or d == 0:
        return None, None, None, None, None

    # Validate k.
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        return None, None, None, None, None

    # Validate iterations.
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations <= 0
    ):
        return None, None, None, None, None

    # Validate tol.
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    # Validate verbose.
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Initialize the model variables.
    pi, m, S = initialize(X, k)

    # Check the initialization output.
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # Run the initial expectation step.
    g, l = expectation(X, pi, m, S)

    # Check the initial expectation output.
    if g is None or l is None:
        return None, None, None, None, None

    # Display the initial likelihood when requested.
    if verbose:
        print(f"Log Likelihood after 0 iterations: {l:.5f}")

    # Run each EM update.
    for iteration in range(1, iterations + 1):
        # Save the current likelihood.
        previous_l = l

        # Update the model variables.
        pi, m, S = maximization(X, g)

        # Check the maximization output.
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Recalculate the probabilities and likelihood.
        g, l = expectation(X, pi, m, S)

        # Check the expectation output.
        if g is None or l is None:
            return None, None, None, None, None

        # Calculate the likelihood change.
        difference = abs(l - previous_l)

        # Determine whether the algorithm has converged.
        converged = difference <= tol

        # Display each tenth step and the final step.
        if (
            verbose
            and (
                iteration % 10 == 0
                or converged
                or iteration == iterations
            )
        ):
            print(
                f"Log Likelihood after {iteration} "
                f"iterations: {l:.5f}"
            )

        # Stop when the likelihood change is small enough.
        if converged:
            break

    # Return the completed model.
    return pi, m, S, g, l
