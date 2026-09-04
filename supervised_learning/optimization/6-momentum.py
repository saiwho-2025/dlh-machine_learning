#!/usr/bin/env python3
"""creates a momentum optimizer"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Creates a momentum optimizer."""
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
    return optimizer
