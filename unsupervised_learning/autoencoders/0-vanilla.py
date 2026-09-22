#!/usr/bin/env python3
"""this module creates a vanille autoencoder"""
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """an autoencoder that returns encoder, decorder and auto
        arguments:
        input_dims: an integer containing the dimensions of the model input
        hidden_layers: a list containing the number of nodes for each hidden layer in the encorder
        latent_dims: an integer containing the dimensions of the latent space representation
        returns:
        encoder: the encoder model
        decoder: the decoder model
            auto: the full autoencoder model    """
    # Encoder
    inputs = K.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = K.layers.Dense(nodes, activation='relu')(x)

    latent = K.layers.Dense(latent_dims, activation='relu')(x)
    encoder = K.Model(inputs=inputs, outputs=latent)
  
    # Decoder
    latent_inputs = K.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = K.layers.Dense(nodes, activation='relu')(x)

    outputs = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(inputs=latent_inputs, outputs=outputs)

    # Full autoencoder
    auto_outputs = decoder(encoder(inputs))
    auto = K.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
