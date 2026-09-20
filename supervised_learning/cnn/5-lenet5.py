#!/usr/bin/env python3
"""Modified LeNet-5 architecture using Keras."""

from tensorflow import keras as K


def lenet5(X):
    """Build and compile a modified LeNet-5 model."""
    initializer = K.initializers.HeNormal(seed=0)

    # First convolutional and pooling layers.
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Second convolutional and pooling layers.
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten the convolutional output.
    flat = K.layers.Flatten()(pool2)

    # Fully connected layers.
    dense1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flat)

    dense2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(dense1)

    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(dense2)

    # Build and compile the model.
    model = K.Model(inputs=X, outputs=output)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
