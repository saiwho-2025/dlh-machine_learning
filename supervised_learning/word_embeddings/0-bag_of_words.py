#!/usr/bin/env python3
"""Build bag-of-words embeddings using NumPy."""

import re
from collections import Counter

import numpy as np


def bag_of_words(sentences, vocab=None):
    """Return a word-count matrix and its column labels.

    Match words case-insensitively. Use the supplied vocabulary order,
    or sort the observed words alphabetically when building a vocabulary.
    """
    counts = [
        Counter(re.findall(r"\w+", sentence.lower()))
        for sentence in sentences
    ]

    if vocab is None:
        features = sorted({word for row in counts for word in row})
    else:
        features = list(vocab)

    keys = [feature.lower() for feature in features]
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    for i, row in enumerate(counts):
        embeddings[i] = [row[word] for word in keys]
