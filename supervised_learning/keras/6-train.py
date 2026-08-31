#!/usr/bin/env python3
"""Trains a neural network with validation and early stopping."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Trains the model and returns the History object."""

    # Start with an empty list of callbacks.
    callbacks = []

    # Early stopping only makes sense if validation data is provided.
    if early_stopping and validation_data is not None:

        # Stop training when validation loss stops improving.
        # patience specifies how many epochs to wait for improvement
        # before stopping.
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )

        # Add the early stopping callback to the list.
        callbacks.append(early_stop)

    # Train the model.
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    # Return the History object containing the training results.
    return history
