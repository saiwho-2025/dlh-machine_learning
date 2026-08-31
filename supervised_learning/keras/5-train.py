#!/usr/bin/env python3
"""Trains a neural network and analyzes validation data."""


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Trains the model and returns the History object."""

    # Train the model using mini-batch gradient descent.
    #
    # validation_data, when provided, is used to evaluate the model
    # after each epoch without updating the model's weights.
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )

    # Return the History object containing training and validation
    # metrics such as loss and accuracy.
    return history
