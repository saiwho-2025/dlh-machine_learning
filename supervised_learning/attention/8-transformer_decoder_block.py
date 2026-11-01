#!/usr/bin/env python3
"""Decoder block for a transformer"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Creates a decoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        Args:
            dm: int, dimensionality of the model
            h: int, number of heads
            hidden: int, number of hidden units in the fully connected layer
            drop_rate: float, the dropout rate
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        Args:
            x: tensor of shape (batch, target_seq_len, dm) containing the
               input to the decoder block
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                            containing the output of the encoder
            training: boolean, whether the model is in training mode
            look_ahead_mask: mask applied to the first multi head attention
            padding_mask: mask applied to the second multi head attention

        Returns:
            tensor of shape (batch, target_seq_len, dm) containing the
            block's output
        """
        # Sub-layer 1: masked self-attention -> dropout -> add & norm
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # Sub-layer 2: encoder-decoder attention -> dropout -> add & norm
        # Queries come from the decoder; keys and values from the encoder
        attn2, _ = self.mha2(out1, encoder_output, encoder_output,
                             padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Sub-layer 3: feed forward -> dropout -> add & norm
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
