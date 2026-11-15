import numpy as np

from connpre.diagnostics import site_decoding_accuracy, stratified_folds


def blobs(per_site=30, n_sites=4, dim=10, shift=3.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_sites, dim)) * shift
    site = np.repeat([f"s{k}" for k in range(n_sites)], per_site)
    x = centers[np.repeat(np.arange(n_sites), per_site)] + rng.standard_normal(
        (n_sites * per_site, dim)
    )
    return x, site


def test_stratified_folds_partition_and_keep_every_class_in_each_test_fold():
    labels = np.repeat(np.arange(4), [20, 15, 10, 7])
    seen = []
    for tr, te in stratified_folds(labels, 5, seed=0):
        assert len(np.intersect1d(tr, te)) == 0
        assert set(labels[te]) == {0, 1, 2, 3}
        seen.append(te)
    assert sorted(np.concatenate(seen).tolist()) == list(range(len(labels)))


def test_separated_sites_are_decoded_and_shuffled_sites_are_not():
    x, site = blobs()
    gram = x @ x.T
    good = site_decoding_accuracy(gram, site)
    assert good["accuracy"] > 0.95
    assert good["majority_share"] == 0.25 and good["n_sites"] == 4
    shuffled = np.random.default_rng(1).permutation(site)
    bad = site_decoding_accuracy(gram, shuffled)
    assert bad["accuracy"] < 0.5


def test_decoding_is_deterministic_for_a_seed():
    x, site = blobs(seed=2)
    gram = x @ x.T
    a = site_decoding_accuracy(gram, site, seed=3)
    b = site_decoding_accuracy(gram, site, seed=3)
    assert a == b
