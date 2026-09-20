#!/usr/bin/env python3
"""This module rotates an image 90 degrees counter-clockwise."""

import tensorflow as tf


def rotate_image(image):
    """Rotate an image 90 degrees counter-clockwise."""
    return tf.image.rot90(image)
