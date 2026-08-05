#!/usr/bin/env python3
"""Run EM on a Gaussian mixture model."""

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
    Run expectation maximization on a GMM.

    Args:
        X: NumPy array with shape (n, d).
        k: Positive cluster count.
        iterations: Maximum update count.
        tol: Nonnegative convergence tolerance.
        verbose: Enables progress output.

    Returns:
        Model parameters, responsibilities, and log likelihood.
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

    # Initialize the model parameters.
    pi, m, S = initialize(X, k)

    # Check the initialization result.
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # Calculate the initial expectation step.
    g, log_likelihood = expectation(X, pi, m, S)

    # Check the expectation result.
    if g is None or log_likelihood is None:
        return None, None, None, None, None

    # Display the initial value when requested.
    if verbose:
        print(
            "Log Likelihood after 0 iterations: "
            f"{log_likelihood:.5f}"
        )

    # Execute the EM updates.
    for iteration in range(1, iterations + 1):
        # Save the current log likelihood.
        previous_log_likelihood = log_likelihood

        # Update the model parameters.
        pi, m, S = maximization(X, g)

        # Check the maximization result.
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Recalculate responsibilities and likelihood.
        g, log_likelihood = expectation(X, pi, m, S)

        # Check the expectation result.
        if g is None or log_likelihood is None:
            return None, None, None, None, None

        # Measure the likelihood change.
        difference = abs(
            log_likelihood - previous_log_likelihood
        )

        # Check convergence.
        converged = difference <= tol

        # Display each tenth update and the final update.
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
                f"iterations: {log_likelihood:.5f}"
            )

        # Stop at convergence.
        if converged:
            break

    # Keep the return value compatible with NumPy rounding.
    log_likelihood = np.float64(log_likelihood)

    # Return the completed model.
    return pi, m, S, g, log_likelihood
