"""Diagnostics: how well can the site be read from the features."""

from __future__ import annotations

import numpy as np

from connpre.ridge import ridge_path


def stratified_folds(labels: np.ndarray, n_splits: int, seed: int):
    """Yield (train, test) index pairs with every class spread over the folds."""
    rng = np.random.default_rng(seed)
    assign = np.empty(len(labels), dtype=np.int64)
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c)
        rng.shuffle(pos)
        assign[pos] = (np.arange(len(pos)) + rng.integers(n_splits)) % n_splits
    for k in range(n_splits):
        yield np.flatnonzero(assign != k), np.flatnonzero(assign == k)


def site_decoding_accuracy(
    gram: np.ndarray,
    site: np.ndarray,
    rel_alpha: float = 1.0,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Cross-validated accuracy of ridge regression on one-hot site labels.

    The ridge strength is fixed in advance (relative to the kernel scale).
    """
    names, labels = np.unique(site, return_inverse=True)
    onehot = np.eye(len(names))[labels]
    hits = np.zeros(len(site), dtype=bool)
    for tr, te in stratified_folds(labels, n_splits, seed):
        path = ridge_path(
            gram[np.ix_(tr, tr)],
            onehot[tr],
            gram[np.ix_(te, tr)],
            np.array([rel_alpha]),
        )
        hits[te] = path[0].argmax(axis=1) == labels[te]
    counts = np.bincount(labels)
    return {
        "accuracy": float(hits.mean()),
        "majority_share": float(counts.max() / len(site)),
        "n_sites": int(len(names)),
        "n_subjects": int(len(site)),
    }
