#!/usr/bin/env python3
"""this module creates a convolutional autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input.
        filters (list): Number of filters for each encoder convolution.
        latent_dims (tuple): Dimensions of the latent representation.

    Returns:
        encoder: Encoder model.
        decoder: Decoder model.
        auto: Full convolutional autoencoder model.
    """

    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)

        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(inputs=inputs, outputs=x)

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs

    reversed_filters = filters[::-1]

    # Decoder convolutions except the second-to-last and last
    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)

        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Second-to-last convolution
    x = keras.layers.Conv2D(
        filters=reversed_filters[-1],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)

    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Last convolution
    outputs = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # Full autoencoder
    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
