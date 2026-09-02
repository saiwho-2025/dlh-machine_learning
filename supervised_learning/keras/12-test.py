#!/usr/bin/env python3
"""Tests a neural network."""


def test_model(network, data, labels, verbose=True):
    """Tests the network using the given data and labels."""
    loss, accuracy = network.evaluate(
        data,
        labels,
        verbose=verbose
    )

    return loss, accuracy
