#!/usr/bin/env python3
"""Trains a neural network with validation, callbacks, and model saving."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """Trains the model and returns the History object."""

    # Store all callbacks that will be used during training.
    callbacks = []

    # Add early stopping if requested and validation data exists.
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )

        callbacks.append(early_stop)

    # Add learning rate decay if requested and validation data exists.
    if learning_rate_decay and validation_data is not None:

        # Inverse time decay:
        # learning_rate = alpha / (1 + decay_rate * epoch)
        def schedule(epoch, learning_rate):
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(
            schedule,
            verbose=1
        )

        callbacks.append(lr_decay)

    # Save the model whenever validation loss reaches a new minimum.
    if save_best and filepath is not None and validation_data is not None:

        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )

        callbacks.append(checkpoint)

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

    # Return the History object containing the training results.
    return history
