"""Folds, nested site-stratified label subsets and calibration sets."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

N_ALL = 10**9


@dataclass(frozen=True)
class Fold:
    """One train/test split; indices point into the cohort."""

    name: str
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Draw:
    """Calibration subjects and a nested order of labelled subjects (disjoint)."""

    cal: np.ndarray
    order: np.ndarray

    def labelled(self, n: int) -> np.ndarray:
        return self.order[: min(n, len(self.order))]


def large_sites(site: np.ndarray, min_site: int) -> list[str]:
    """Sites with at least min_site subjects, largest first (ties by name)."""
    names, counts = np.unique(site, return_counts=True)
    keep = [(-c, str(s)) for s, c in zip(names, counts, strict=True) if c >= min_site]
    return [s for _, s in sorted(keep)]


def loso_folds(site: np.ndarray, min_site: int) -> list[Fold]:
    """Leave one larger site out; every other site (small ones too) trains."""
    folds = []
    for s in large_sites(site, min_site):
        test = np.flatnonzero(site == s)
        train = np.flatnonzero(site != s)
        folds.append(Fold(s, train, test))
    return folds


def insite_folds(site: np.ndarray, n_splits: int, seed: int) -> list[Fold]:
    """Random folds stratified by site: every site is in training and in test."""
    rng = np.random.default_rng([seed, 7])
    assign = np.empty(len(site), dtype=np.int64)
    for s in np.unique(site):
        pos = np.flatnonzero(site == s)
        rng.shuffle(pos)
        assign[pos] = (np.arange(len(pos)) + rng.integers(n_splits)) % n_splits
    return [
        Fold(f"split{k}", np.flatnonzero(assign != k), np.flatnonzero(assign == k))
        for k in range(n_splits)
    ]


def stratified_order(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions ordered so that every prefix is close to proportional by group."""
    keys = np.empty(len(groups))
    for g in np.unique(groups):
        pos = np.flatnonzero(groups == g)
        rng.shuffle(pos)
        keys[pos] = (np.arange(len(pos)) + rng.random(len(pos))) / len(pos)
    return np.argsort(keys, kind="stable")


def draw_sets(
    train: np.ndarray, site: np.ndarray, n_cal: int, seed: int, fold_name: str
) -> Draw:
    """The first n_cal of a stratified order calibrate; the rest is the label order."""
    salt = zlib.crc32(fold_name.encode())
    rng = np.random.default_rng([seed, salt])
    order = train[stratified_order(site[train], rng)]
    return Draw(cal=order[:n_cal], order=order[n_cal:])


def inner_split(
    train: np.ndarray, site: np.ndarray, seed: int, fold_name: str, loso: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Pool for labelled subjects and validation subjects, both inside the training set.

    Leave-one-site-out folds validate on the largest training site; other folds on a
    stratified quarter of the training subjects.
    """
    if loso:
        names, counts = np.unique(site[train], return_counts=True)
        big = names[np.argmax(counts)]
        val = train[site[train] == big]
        return train[site[train] != big], val
    salt = zlib.crc32(fold_name.encode())
    rng = np.random.default_rng([seed, salt, 3])
    order = train[stratified_order(site[train], rng)]
    n_val = len(train) // 4
    return order[n_val:], order[:n_val]
