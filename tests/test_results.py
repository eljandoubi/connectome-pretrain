import numpy as np

from connpre.results import RunStore, keys_of, load_mode, pooled, write_rows_csv


def make_store(fold, n_test, offset):
    rng = np.random.default_rng(offset)
    y = {"age": rng.normal(15, 4, n_test), "fiq": rng.normal(105, 12, n_test)}
    store = RunStore(
        "loso",
        fold,
        np.arange(n_test) + offset,
        np.arange(n_test) + 1000 * offset,
        [fold] * n_test,
        y,
    )
    for seed in range(3):
        for target in ("age", "fiq"):
            pred = y[target] + rng.normal(0, 2, n_test)
            y_cal = rng.normal(0, 3, 10)
            cal = (y_cal, y_cal + rng.normal(0, 2, 10))
            store.add("ridge", target, 100, seed, pred, cal=cal, alpha=1.0)
            store.add("oracle", target, 100, seed, y[target] * 0 + y[target].mean())
    return store


def test_save_load_round_trip_and_pooling(tmp_path):
    a, b = make_store("A", 12, 1), make_store("B", 8, 2)
    a.save(str(tmp_path))
    b.save(str(tmp_path))
    folds = load_mode(str(tmp_path), "loso")
    assert [f["fold"] for f in folds] == ["A", "B"]
    assert keys_of(folds) == [
        ("oracle", "age", 100),
        ("oracle", "fiq", 100),
        ("ridge", "age", 100),
        ("ridge", "fiq", 100),
    ]
    pool = pooled(folds, "ridge", "age", 100)
    assert pool["y"].shape == (20,)
    assert pool["p"].shape == (3, 20)
    assert pool["q"].shape == (3, 20, 2)
    assert list(pool["site"]) == ["A"] * 12 + ["B"] * 8
    assert np.allclose(pool["y"][:12], a.y["age"])
    assert np.allclose(pool["p"][1, :12], a._pred[("ridge", "age", 100)][1], atol=1e-5)
    assert np.isfinite(pool["q"][:, :, 1]).all()
    assert np.isnan(pooled(folds, "oracle", "age", 100)["q"]).all()


def test_rows_carry_scores_and_extra_fields(tmp_path):
    store = make_store("A", 12, 1)
    row = store.rows[0]
    assert row["arm"] == "ridge" and row["alpha"] == 1.0
    assert {"r2", "mae", "spearman", "cover90", "width90"} <= set(row)
    path = tmp_path / "rows.csv"
    write_rows_csv(store.rows, str(path))
    header = path.read_text().splitlines()[0].split(",")
    assert "cover90" in header and "alpha" in header
    assert len(path.read_text().splitlines()) == 1 + len(store.rows)
