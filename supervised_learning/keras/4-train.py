#!/usr/bin/env python3
"""Trains a neural network using mini-batch gradient descent."""

def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Trains the model and returns the History object."""

    # Train the model using mini-batch gradient descent.
    #
    # batch_size determines how many samples are processed
    # before the model's weights are updated.
    #
    # epochs determines how many times the entire dataset
    # is passed through the network.
    #
    # verbose controls whether Keras displays training progress.
    #
    # shuffle determines whether the training data is shuffled
    # at the beginning of each epoch.
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    # fit() returns a History object containing information
    # collected during training, such as loss and accuracy.
    return history
