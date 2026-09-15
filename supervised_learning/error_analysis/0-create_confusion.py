#!/usr/bin/env python3
"""Creates a confusion matrix from one-hot encoded labels and predictions."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    Args:
        labels: One-hot numpy.ndarray of shape (m, classes)
            containing the correct labels.
        logits: One-hot numpy.ndarray of shape (m, classes)
            containing the predicted labels.

    Returns:
        A numpy.ndarray of shape (classes, classes), where rows
        represent the correct labels and columns represent the
        predicted labels.
    """
    # Convert one-hot encoded correct labels to class indices.
    correct = np.argmax(labels, axis=1)

    # Convert one-hot encoded predictions to class indices.
    predicted = np.argmax(logits, axis=1)

    # Get the number of classes from the labels array.
    classes = labels.shape[1]

    # Create an empty confusion matrix.
    confusion = np.zeros((classes, classes), dtype=int)

    # Count each correct/predicted label combination.
    for correct_label, predicted_label in zip(correct, predicted):
        confusion[correct_label, predicted_label] += 1

    return confusion
