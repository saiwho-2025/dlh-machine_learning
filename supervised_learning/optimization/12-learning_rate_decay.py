#!/usr/bin/env python3
"""creates a learning rate decay operation"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Creates an inverse time learning rate decay operation."""
    global_step = tf.Variable(0, trainable=False)

    return tf.compat.v1.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True
    )
