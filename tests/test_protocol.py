import numpy as np

from connpre.protocol import (
    N_ALL,
    draw_sets,
    inner_split,
    insite_folds,
    large_sites,
    loso_folds,
    stratified_order,
)


def make_sites():
    return np.array(["A"] * 50 + ["B"] * 30 + ["C"] * 20 + ["D"] * 7 + ["E"] * 3)


def test_large_sites_sorted_by_size_then_name():
    site = make_sites()
    assert large_sites(site, 20) == ["A", "B", "C"]
    assert large_sites(site, 3) == ["A", "B", "C", "D", "E"]
    assert large_sites(site, 100) == []


def test_loso_folds_hold_out_one_site_and_train_on_the_rest():
    site = make_sites()
    folds = loso_folds(site, 20)
    assert [f.name for f in folds] == ["A", "B", "C"]
    for f in folds:
        assert set(site[f.test]) == {f.name}
        assert f.name not in set(site[f.train])
        assert len(np.intersect1d(f.train, f.test)) == 0
        assert len(f.train) + len(f.test) == len(site)
    tested = np.sort(np.concatenate([f.test for f in folds]))
    assert np.array_equal(tested, np.flatnonzero(np.isin(site, ["A", "B", "C"])))


def test_small_sites_always_train():
    site = make_sites()
    for f in loso_folds(site, 20):
        assert {"D", "E"} <= set(site[f.train])


def test_insite_folds_partition_the_cohort_and_cover_every_site():
    site = make_sites()
    folds = insite_folds(site, n_splits=5, seed=0)
    tested = np.concatenate([f.test for f in folds])
    assert np.array_equal(np.sort(tested), np.arange(len(site)))
    for f in folds:
        assert len(np.intersect1d(f.train, f.test)) == 0
        assert {"A", "B", "C", "D"} <= set(site[f.test])
    sizes = [len(f.test) for f in folds]
    assert max(sizes) - min(sizes) <= 5


def test_stratified_order_prefixes_are_close_to_proportional():
    groups = np.array(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    order = stratified_order(groups, np.random.default_rng(0))
    assert sorted(order.tolist()) == list(range(100))
    for m in (10, 25, 50, 80):
        for g, share in (("a", 0.5), ("b", 0.3), ("c", 0.2)):
            got = int((groups[order[:m]] == g).sum())
            assert abs(got - m * share) <= 2


def test_draw_sets_are_disjoint_deterministic_and_inside_train():
    site = make_sites()
    train = np.flatnonzero(site != "A")
    d1 = draw_sets(train, site, n_cal=20, seed=1, fold_name="A")
    d2 = draw_sets(train, site, n_cal=20, seed=1, fold_name="A")
    d3 = draw_sets(train, site, n_cal=20, seed=2, fold_name="A")
    assert np.array_equal(d1.cal, d2.cal) and np.array_equal(d1.order, d2.order)
    assert not np.array_equal(d1.cal, d3.cal)
    assert len(d1.cal) == 20
    assert len(np.intersect1d(d1.cal, d1.order)) == 0
    assert set(d1.cal) | set(d1.order) == set(train)


def test_labelled_subsets_are_nested_and_stratified():
    site = make_sites()
    train = np.arange(len(site))
    d = draw_sets(train, site, n_cal=10, seed=0, fold_name="x")
    a, b = d.labelled(20), d.labelled(40)
    assert set(a) <= set(b)
    assert np.array_equal(d.labelled(N_ALL), d.order)
    counts = {s: int((site[b] == s).sum()) for s in "ABC"}
    assert counts["A"] > counts["B"] > counts["C"] > 0


def test_inner_split_by_site_validates_on_the_largest_training_site():
    site = make_sites()
    train = np.flatnonzero(site != "A")
    pool, val = inner_split(train, site, seed=0, fold_name="A", loso=True)
    assert set(site[val]) == {"B"}
    assert "B" not in set(site[pool])
    assert set(pool) | set(val) == set(train)


def test_inner_split_random_keeps_all_sites_in_both_parts():
    site = make_sites()
    train = np.arange(len(site))
    pool, val = inner_split(train, site, seed=0, fold_name="split0", loso=False)
    assert len(np.intersect1d(pool, val)) == 0
    assert set(pool) | set(val) == set(train)
    assert abs(len(val) - len(train) // 4) <= 1
    assert {"A", "B", "C"} <= set(site[val])
