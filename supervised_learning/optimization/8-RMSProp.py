#!/usr/bin/env python3
"""creates an RMSProp optimizer"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Creates an RMSProp optimizer."""
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
