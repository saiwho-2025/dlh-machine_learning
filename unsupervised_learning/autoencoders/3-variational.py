#!/usr/bin/env python3
"""Module for creating a variational autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes in each encoder hidden layer.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: The encoder, decoder, and full autoencoder models.
    """
    # Build the encoder
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(
            nodes,
            activation="relu"
        )(encoded)

    # Mean of the latent distribution
    mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    # Log variance of the latent distribution
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

    # Generate the latent representation
    latent = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([mean, log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[latent, mean, log_var]
    )

    # Build the decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(
            nodes,
            activation="relu"
        )(decoded)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation="sigmoid"
    )(decoded)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    # Build the full autoencoder by chaining the models
    encoded_output = encoder(encoder_input)[0]
    auto_output = decoder(encoded_output)

    auto = keras.Model(
        inputs=encoder_input,
        outputs=auto_output
    )

    # Compile the autoencoder
    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
