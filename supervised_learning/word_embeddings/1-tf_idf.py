#!/usr/bin/env python3
"""Create TF-IDF embeddings using NumPy."""

import re
from collections import Counter

import numpy as np


def tf_idf(sentences, vocab=None):
    """Return TF-IDF embeddings and a NumPy array of feature names.

    Match words of at least two characters case-insensitively.
    Preserve the supplied vocabulary order, or sort observed words.
    Use smoothed IDF weights and L2-normalize each nonzero row.
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
    embeddings = np.zeros((len(sentences), len(features)), dtype=float)
    for i, row in enumerate(counts):
        embeddings[i] = [row[word] for word in keys]

    document_frequencies = np.count_nonzero(embeddings, axis=0)
    idf = np.log((1 + len(sentences)) / (1 + document_frequencies)) + 1
    embeddings *= idf

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.divide(embeddings, norms, out=embeddings, where=norms != 0)

    return embeddings, np.array(features, dtype=str)
