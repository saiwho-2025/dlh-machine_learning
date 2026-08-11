#!/usr/bin/env python3
"""this model  builds a neural network with the keras library"""

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """the function builds a neural network with the Keras library"""
    