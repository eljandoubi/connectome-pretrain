"""How well is the site decodable from the features the models see?

usage: python scripts/site_decoding.py --data data/abide_cc200.npz \
    --out results/site_decoding.json
"""

import argparse
import json
from pathlib import Path

from connpre.baselines import gram_matrix
from connpre.data import lag1_autocorr, load_cohort
from connpre.diagnostics import site_decoding_accuracy
from connpre.features import (
    TangentSpace,
    correlation_vectors,
    ledoit_wolf_covs,
)
from connpre.settings import TANGENT_TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cohort = load_cohort(args.data)
    site = cohort.site
    covs = ledoit_wolf_covs(cohort.x)
    tangent = TangentSpace(tol=TANGENT_TOL).fit(covs).transform(covs)
    lag = lag1_autocorr(cohort.x)[:, None]
    lag = (lag - lag.mean()) / lag.std()
    result = {
        "correlation": site_decoding_accuracy(
            gram_matrix(correlation_vectors(cohort.x)), site
        ),
        "tangent": site_decoding_accuracy(gram_matrix(tangent), site),
        "lag1_autocorrelation_only": site_decoding_accuracy(gram_matrix(lag), site),
    }
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
