#!/usr/bin/env python3
"""This module randomly crops an image."""

import tensorflow as tf


def crop_image(image, size):
    """Randomly crop an image to the given size."""
    return tf.image.random_crop(image, size=size)
