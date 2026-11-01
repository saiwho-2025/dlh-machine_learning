#!/usr/bin/env python3
"""Multi-head attention layer"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention"""

    def __init__(self, dm, h):
        """
        Class constructor

        Args:
            dm: int, dimensionality of the model
            h: int, number of heads (dm is divisible by h)
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch):
        """
        Splits the last dimension into (h, depth) and transposes so the
        result has shape (batch, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Args:
            Q: tensor of shape (batch, seq_len_q, dk), input for the queries
            K: tensor of shape (batch, seq_len_v, dk), input for the keys
            V: tensor of shape (batch, seq_len_v, dv), input for the values
            mask: always None

        Returns:
            output, weights
            output: tensor of shape (batch, seq_len_q, dm) containing the
                    multi head attention result
            weights: tensor of shape (batch, h, seq_len_q, seq_len_v)
                     containing the attention weights
        """
        batch = tf.shape(Q)[0]

        # Linear projections: (batch, seq_len, dm)
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # Split into heads: (batch, h, seq_len, depth)
        q = self.split_heads(q, batch)
        k = self.split_heads(k, batch)
        v = self.split_heads(v, batch)

        # Scaled dot product attention on every head in parallel
        # attention: (batch, h, seq_len_q, depth)
        # weights:   (batch, h, seq_len_q, seq_len_v)
        attention, weights = sdp_attention(q, k, v, mask)

        # Concat heads: (batch, seq_len_q, h, depth) -> (batch, seq_len_q, dm)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(attention, (batch, -1, self.dm))

        # Final linear layer
        output = self.linear(concat)

        return output, weights
