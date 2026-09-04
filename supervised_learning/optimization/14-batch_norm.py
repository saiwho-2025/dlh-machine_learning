#!/usr/bin/env python3
"""normalize in tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    # Create the dense layer with the specified kernel initializer
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg')
    )(prev)
    
    # Calculate mean and variance for the batch
    mean, variance = tf.nn.moments(dense_layer, axes=[0])
    
    # Define trainable scaling (gamma) and shifting (beta) parameters
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    
    # Apply batch normalization with epsilon = 1e-7
    normalized = tf.nn.batch_normalization(
        x=dense_layer,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-7
    )
    
    # Apply the activation function to the output
    if activation is not None:
        return activation(normalized)
    return normalized
