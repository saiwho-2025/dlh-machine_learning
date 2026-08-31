#!/usr/bin/env python3
"""Converts a label vector into a one-hot matrix."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix."""

    # Convert the labels into a one-hot encoded matrix.
    # classes specifies the total number of possible classes.
    one_hot_labels = K.utils.to_categorical(labels, classes)

    # Return the one-hot matrix.
    return one_hot_labels
