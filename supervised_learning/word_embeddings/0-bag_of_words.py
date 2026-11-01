#!/usr/bin/env python3
"""Build bag-of-words embeddings using NumPy."""

import re
from collections import Counter

import numpy as np


def bag_of_words(sentences, vocab=None):
    """Return word counts and a NumPy array of feature names.

    Match words of at least two characters case-insensitively.
    Preserve the supplied vocabulary order, or sort observed words.
    """
    counts = [
        Counter(re.findall(r"\b\w\w+\b", sentence.lower()))
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

    return embeddings, np.array(features, dtype=str)
