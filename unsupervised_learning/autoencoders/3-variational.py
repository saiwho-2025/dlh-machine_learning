#!/usr/bin/env python3
"""Module for creating a variational autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Nodes in each encoder hidden layer.
        latent_dims (int): Dimensions of the latent representation.

    Returns:
        tuple: The encoder, decoder, and autoencoder models.
    """
    # Create encoder input
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    # Create encoder hidden layers
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(
            nodes,
            activation="relu"
        )(encoded)

    # Create mean and log variance layers
    mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    def sampling(args):
        """Sample from the latent distribution."""
        mean, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mean)
        )
        return mean + keras.backend.exp(log_var / 2) * epsilon

    # Create latent representation
    latent = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([mean, log_var])

    # Create encoder model
    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[latent, mean, log_var]
    )

    # Create decoder input
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    # Create decoder hidden layers
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(
            nodes,
            activation="relu"
        )(decoded)

    # Create decoder output
    decoder_output = keras.layers.Dense(
        input_dims,
        activation="sigmoid"
    )(decoded)

    # Create decoder model
    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    # Create full autoencoder
    encoded_output = encoder(encoder_input)[0]
    auto_output = decoder(encoded_output)

    auto = keras.Model(
        inputs=encoder_input,
        outputs=auto_output
    )

    # Compile autoencoder
    auto.compile(
        optimizer="adam",
        loss=keras.losses.binary_crossentropy
    )

    return encoder, decoder, auto
