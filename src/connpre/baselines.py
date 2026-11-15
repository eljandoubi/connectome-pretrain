"""Baseline arms: correlation ridge, tangent ridge, tangent kernel ridge, site means."""

from __future__ import annotations

import numpy as np

from connpre.data import Cohort
from connpre.features import TangentSpace
from connpre.protocol import Fold, draw_sets
from connpre.results import RunStore
from connpre.ridge import GAMMA_MULTS, REL_ALPHAS, rbf_cv_predict, ridge_cv_predict
from connpre.settings import TANGENT_TOL, TARGETS


def gram_matrix(features: np.ndarray) -> np.ndarray:
    a = features.astype(np.float64)
    return a @ a.T


def new_store(cohort: Cohort, fold: Fold, mode: str) -> RunStore:
    return RunStore(
        mode,
        fold.name,
        fold.test,
        cohort.sub_id[fold.test],
        cohort.site[fold.test],
        {t: cohort.target(t)[fold.test] for t in TARGETS},
    )


def site_mean_prediction(lab_site, y_lab, target_site) -> np.ndarray:
    """Mean label of the labelled subjects of each target subject's own site."""
    out = np.full(len(target_site), y_lab.mean())
    for s in np.unique(target_site):
        m = lab_site == s
        if m.any():
            out[target_site == s] = y_lab[m].mean()
    return out


def fold_baselines(
    cohort: Cohort,
    fold: Fold,
    covs: np.ndarray,
    corr: np.ndarray,
    mode: str,
    seeds: tuple,
    sizes: tuple,
    n_cal: int,
) -> RunStore:
    """Run every baseline arm on one fold for all seeds, label sizes and targets."""
    site = cohort.site
    y_all = np.stack([cohort.target(t) for t in TARGETS], axis=1)
    tangent = TangentSpace(tol=TANGENT_TOL).fit(covs[fold.train])
    g_tan = gram_matrix(tangent.transform(covs))
    g_corr = gram_matrix(corr)
    store = new_store(cohort, fold, mode)
    store.extra["tangent_iterations"] = int(tangent.n_iter_)
    n_te = len(fold.test)
    for seed in seeds:
        draw = draw_sets(fold.train, site, n_cal, seed, fold.name)
        cal = draw.cal
        both = np.concatenate([cal, fold.test])
        for n in sizes:
            lab = draw.labelled(n)
            ridge_args = (y_all, lab, both, site[lab])
            arms = {
                "corr_ridge": ridge_cv_predict(g_corr, *ridge_args),
                "tangent_ridge": ridge_cv_predict(g_tan, *ridge_args),
                "tangent_krr": rbf_cv_predict(g_tan, *ridge_args, ref=fold.train),
            }
            for arm, (pred, chosen) in arms.items():
                for t, target in enumerate(TARGETS):
                    extra = {}
                    if arm == "tangent_krr":
                        extra["gamma_mult"] = float(GAMMA_MULTS[chosen[t, 0]])
                        extra["rel_alpha"] = float(REL_ALPHAS[chosen[t, 1]])
                    else:
                        extra["rel_alpha"] = float(REL_ALPHAS[chosen[t]])
                    cal_pair = (y_all[cal, t], pred[: len(cal), t])
                    store.add(
                        arm, target, n, seed, pred[len(cal) :, t], cal_pair, **extra
                    )
            for t, target in enumerate(TARGETS):
                const = float(y_all[lab, t].mean())
                cal_pair = (y_all[cal, t], np.full(len(cal), const))
                store.add("site_const", target, n, seed, np.full(n_te, const), cal_pair)
                if mode == "loso":
                    oracle = float(y_all[fold.test, t].mean())
                    store.add("site_oracle", target, n, seed, np.full(n_te, oracle))
                else:
                    own = site_mean_prediction(
                        site[lab], y_all[lab, t], site[fold.test]
                    )
                    own_cal = site_mean_prediction(site[lab], y_all[lab, t], site[cal])
                    store.add(
                        "site_mean", target, n, seed, own, (y_all[cal, t], own_cal)
                    )
    return store
