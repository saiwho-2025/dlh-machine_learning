#!/usr/bin/env python3
"""Calculate a GMM using Scikit-learn."""

# Import the only permitted module.
import sklearn.mixture


def gmm(X, k):
    """
    Calculate a Gaussian mixture model.

    Args:
        X: Dataset with shape (n, d).
        k: Number of Gaussian components.

    Returns:
        pi, m, S, clss, and bic.
    """
    # Create a full-covariance GMM.
    model = sklearn.mixture.GaussianMixture(
        n_components=k
    )

    # Train the model using the dataset.
    model.fit(X)

    # Extract the cluster priors.
    pi = model.weights_

    # Extract the centroid means.
    m = model.means_

    # Extract the covariance matrices.
    S = model.covariances_

    # Predict each data point's cluster index.
    clss = model.predict(X)

    # Calculate the model's BIC value.
    bic = model.bic(X)

    # Return all calculated values.
    return pi, m, S, clss, bic
