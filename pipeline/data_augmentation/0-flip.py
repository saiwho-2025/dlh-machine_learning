#!/usr/bin/env python3
"""This module flips an image horizontally from left to right."""

import tensorflow as tf


def flip_image(image):
    """Flip an image horizontally."""
    return tf.image.flip_left_right(image)
