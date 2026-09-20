#!/usr/bin/env python3
"""this module flip an image horizontally from left to right."""


def flip_image(image):
    return tf.image.flip_left_right(image)