#!/usr/bin/env python3
"""this module calculates the scaled dot product attention"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Args:
        Q: tensor of shape (..., seq_len_q, dk) containing the query matrix
        K: tensor of shape (..., seq_len_v, dk) containing the key matrix
        V: tensor of shape (..., seq_len_v, dv) containing the value matrix
        mask: optional tensor broadcastable to (..., seq_len_q, seq_len_v)

    Returns:
        output, weights
        output: tensor of shape (..., seq_len_q, dv) containing the
                scaled dot product attention
        weights: tensor of shape (..., seq_len_q, seq_len_v) containing
                 the attention weights
    """
    # MatMul: Q . K^T -> (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale by sqrt(dk)
    dk = tf.cast(tf.shape(K)[-1], matmul_qk.dtype)
    scaled = matmul_qk / tf.math.sqrt(dk)

    # Mask (optional): multiply by -1e9 and add to the scaled matrix
    if mask is not None:
        scaled += mask * -1e9

    # SoftMax over the last axis (keys)
    weights = tf.nn.softmax(scaled, axis=-1)

    # MatMul: weights . V -> (..., seq_len_q, dv)
    output = tf.matmul(weights, V)

    return output, weights
