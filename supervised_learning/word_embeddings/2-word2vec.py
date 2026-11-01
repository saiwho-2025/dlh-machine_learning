#!/usr/bin/env python3
"""Create, build, and train a Gensim Word2Vec model."""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Return a trained Word2Vec model.

    Each sentence must be a list of word tokens.
    Use CBOW when cbow is True and Skip-gram when it is False.
    """
    model = Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )
    return model
