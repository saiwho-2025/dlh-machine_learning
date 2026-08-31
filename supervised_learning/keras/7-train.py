#!/usr/bin/env python3
"""Trains a neural network with validation, early stopping,
and learning rate decay."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """Trains the model and returns the History object."""

    # Store callbacks that will be used during training.
    callbacks = []

    # Early stopping only works when validation data exists.
    if early_stopping and validation_data is not None:

        # Stop training when validation loss stops improving.
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )

        callbacks.append(early_stop)

    # Learning rate decay only works when validation data exists.
    if learning_rate_decay and validation_data is not None:

        # Define the inverse time decay function.
        #
        # alpha is the initial learning rate.
        # decay_rate controls how quickly the learning rate decreases.
        #
        # The learning rate becomes:
        # alpha / (1 + decay_rate * epoch)
        def schedule(epoch, learning_rate):
            return alpha / (1 + decay_rate * epoch)

        # Update the learning rate after each epoch.
        # verbose=1 makes Keras print the new learning rate.
        learning_rate_decay_callback = K.callbacks.LearningRateScheduler(
            schedule,
            verbose=1
        )

        callbacks.append(learning_rate_decay_callback)

    # Train the model using the selected callbacks.
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

    # Return the History object containing training metrics.
    return history
