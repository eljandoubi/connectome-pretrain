"""Download ABIDE I CC200 series with nilearn, filter, z-score and cache one npz.

usage: python scripts/fetch_data.py --out data/abide_cc200.npz [--t0 116] [--lengths]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from connpre.data import build_cohort, save_cohort, site_table

ROOT = Path(__file__).resolve().parents[1]


def _text(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def load_abide(data_dir: str):
    """Series (list of T x 200 arrays) and phenotypic table of the checked subjects."""
    import pandas as pd
    from nilearn import datasets

    bunch = datasets.fetch_abide_pcp(
        data_dir=data_dir,
        derivatives=["rois_cc200"],
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        quality_checked=True,
        verbose=0,
    )
    pheno = pd.DataFrame(bunch.phenotypic)
    series = [np.asarray(a, dtype=np.float64) for a in bunch.rois_cc200]
    table = {
        "site": [_text(v) for v in pheno["SITE_ID"]],
        "age": pd.to_numeric(pheno["AGE_AT_SCAN"], errors="coerce").to_numpy(),
        "fiq": pd.to_numeric(pheno["FIQ"], errors="coerce").to_numpy(),
        "dx": pd.to_numeric(pheno["DX_GROUP"], errors="coerce").to_numpy(),
        "sex": pd.to_numeric(pheno["SEX"], errors="coerce").to_numpy(),
        "sub_id": pd.to_numeric(pheno["SUB_ID"], errors="coerce").to_numpy(),
    }
    return series, table


def print_lengths(series, table):
    """Downloaded subjects per site with valid FIQ and the original series length."""
    sites = np.array(table["site"])
    lengths = np.array([s.shape[0] for s in series])
    fiq = np.asarray(table["fiq"], dtype=float)
    print("site n n_fiq len_min len_med")
    for s in sorted(set(sites), key=lambda s: -(sites == s).sum()):
        m = sites == s
        n_fiq = int(((fiq > 0) & m).sum())
        print(s, int(m.sum()), n_fiq, int(lengths[m].min()), int(np.median(lengths[m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "abide_cc200.npz"))
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "nilearn"))
    ap.add_argument("--t0", type=int, default=116)
    ap.add_argument("--lengths", action="store_true", help="print the length table")
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args()

    series, table = load_abide(args.raw_dir)
    if args.lengths:
        print_lengths(series, table)
    cohort, report = build_cohort(series, table, args.t0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_cohort(args.out, cohort)

    rows = site_table(cohort)
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    report["n_roi"] = int(cohort.x.shape[2])
    report["n_sites"] = len(rows)
    (res / "cohort.json").write_text(json.dumps(report, indent=2) + "\n")
    with open(res / "site_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(report))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
