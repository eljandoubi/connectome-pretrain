"""Per-fold run storage: metric rows as JSON, test predictions as npz."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from connpre.conformal import score_run

NAN2 = (float("nan"), float("nan"))


class RunStore:
    """Collects the runs of one fold: metric rows and held-out predictions."""

    def __init__(self, mode, fold, test_idx, sub_id, site, y):
        self.mode, self.fold = mode, fold
        self.test_idx = np.asarray(test_idx)
        self.sub_id = np.asarray(sub_id)
        self.site = np.asarray(site).astype(str)
        self.y = {k: np.asarray(v, dtype=np.float64) for k, v in y.items()}
        self.rows: list[dict] = []
        self._pred: dict = {}
        self._q: dict = {}
        self.extra: dict = {}

    def add(self, arm, target, n, seed, pred, cal=None, **extra) -> dict:
        """Score one run on the held-out subjects; cal = (y_cal, p_cal) or None."""
        pred = np.asarray(pred, dtype=np.float64)
        score = score_run(self.y[target], pred, *(cal if cal is not None else ()))
        row = {
            "mode": self.mode,
            "fold": self.fold,
            "arm": arm,
            "target": target,
            "n": int(n),
            "seed": int(seed),
            "n_test": len(pred),
            **score,
            **extra,
        }
        self.rows.append(row)
        key = (arm, target, int(n))
        self._pred.setdefault(key, {})[int(seed)] = pred.astype(np.float32)
        q = (row.get("q80", np.nan), row.get("q90", np.nan))
        self._q.setdefault(key, {})[int(seed)] = q
        return row

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{self.mode}-{self.fold}"
        arrays = {
            "test_idx": self.test_idx,
            "sub_id": self.sub_id,
            "site": self.site,
            **{f"y|{k}": v for k, v in self.y.items()},
        }
        for (arm, target, n), by_seed in self._pred.items():
            seeds = sorted(by_seed)
            tag = f"{arm}|{target}|{n}"
            arrays[f"pred|{tag}"] = np.stack([by_seed[s] for s in seeds])
            arrays[f"q|{tag}"] = np.array([self._q[(arm, target, n)][s] for s in seeds])
            arrays[f"seeds|{tag}"] = np.array(seeds)
        np.savez_compressed(d / f"{stem}.npz", **arrays)
        meta = {"mode": self.mode, "fold": self.fold, "rows": self.rows}
        meta["extra"] = self.extra
        (d / f"{stem}.json").write_text(json.dumps(meta) + "\n")


def load_fold(directory: str, mode: str, fold: str) -> dict:
    d = Path(directory)
    meta = json.loads((d / f"{mode}-{fold}.json").read_text())
    with np.load(d / f"{mode}-{fold}.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return {"meta": meta, "arrays": arrays, "fold": fold}


def load_mode(directory: str, mode: str) -> list[dict]:
    names = sorted(
        p.stem[len(mode) + 1 :] for p in Path(directory).glob(f"{mode}-*.json")
    )
    return [load_fold(directory, mode, n) for n in names]


def keys_of(folds: list[dict]) -> list[tuple]:
    """All (arm, target, n) present in every fold."""
    sets = []
    for f in folds:
        found = set()
        for k in f["arrays"]:
            if k.startswith("pred|"):
                arm, target, n = k[5:].split("|")
                found.add((arm, target, int(n)))
        sets.append(found)
    return sorted(set.intersection(*sets)) if sets else []


def pooled(folds: list[dict], arm: str, target: str, n: int) -> dict:
    """Stack the held-out predictions of all folds: y (m,), p (S, m), q (S, m, 2)."""
    ys, ps, qs, sites = [], [], [], []
    for f in folds:
        a = f["arrays"]
        tag = f"{arm}|{target}|{n}"
        p = a[f"pred|{tag}"]
        ys.append(a[f"y|{target}"])
        ps.append(p)
        sites.append(a["site"])
        q = a[f"q|{tag}"]
        qs.append(np.repeat(q[:, None, :], p.shape[1], axis=1))
    return {
        "y": np.concatenate(ys),
        "p": np.concatenate(ps, axis=1),
        "q": np.concatenate(qs, axis=1),
        "site": np.concatenate(sites),
    }


def write_rows_csv(rows: list[dict], path: str) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
