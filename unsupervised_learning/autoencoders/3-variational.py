#!/usr/bin/env python3
"""Module that creates a variational autoencoder."""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes in each hidden layer.
        latent_dims (int): Dimensions of the latent representation.

    Returns:
        tuple: The encoder, decoder, and full autoencoder models.
    """
    # Create the encoder input
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs

    # Add the encoder hidden layers
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(
            nodes,
            activation="relu"
        )(encoded)

    # Create the mean and log variance layers
    mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    # Sample the latent representation
    def sampling(args):
        """Sample a point from the latent distribution."""
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean))
        return mean + K.exp(log_var / 2) * epsilon

    latent = keras.layers.Lambda(sampling)([mean, log_var])

    # Create the encoder model
    encoder = keras.Model(
        inputs=inputs,
        outputs=[latent, mean, log_var]
    )

    # Create the decoder input
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    # Add the decoder hidden layers in reverse order
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(
            nodes,
            activation="relu"
        )(decoded)

    # Create the decoder output layer
    decoded = keras.layers.Dense(
        input_dims,
        activation="sigmoid"
    )(decoded)

    # Create the decoder model
    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoded
    )

    # Create the full variational autoencoder
    auto = keras.Model(
        inputs=inputs,
        outputs=decoder(latent)
    )

    # Compile the full autoencoder
    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
