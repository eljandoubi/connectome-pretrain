"""Longer fine-tuning for age: random initialisation versus the transductive encoder.

usage: python scripts/run_long.py --data D --ckpt K --out runs_long [--folds NYU]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from connpre.baselines import new_store
from connpre.data import load_cohort
from connpre.finetune import FinetuneConfig, finetune, predict
from connpre.model import ModelConfig
from connpre.pretrain import eval_starts
from connpre.protocol import draw_sets, loso_folds
from connpre.settings import MIN_SITE, N_CAL, SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="", help="comma list of fold names")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-cal", type=int, default=N_CAL)
    ap.add_argument("--min-site", type=int, default=MIN_SITE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cohort = load_cohort(args.data)
    x = torch.as_tensor(cohort.x, device=device)
    model_cfg = ModelConfig(n_roi=x.shape[2])
    trans = torch.load(Path(args.ckpt) / "trans.pt", map_location="cpu")
    starts = eval_starts(x.shape[1], model_cfg.window, 4)
    keep = set(filter(None, args.folds.split(",")))
    cfg = FinetuneConfig(lr=args.lr, steps=args.steps)
    for fold in loso_folds(cohort.site, args.min_site):
        if keep and fold.name not in keep:
            continue
        if (Path(args.out) / f"loso-{fold.name}.json").exists():
            continue
        started = time.time()
        store = new_store(cohort, fold, "loso")
        y = cohort.age
        for seed in (int(s) for s in args.seeds.split(",")):
            draw = draw_sets(fold.train, cohort.site, args.n_cal, seed, fold.name)
            lab = draw.labelled(args.n)
            both = np.concatenate([draw.cal, fold.test])
            mu, sd = y[lab].mean(), y[lab].std()
            for arm, state in (("ft_scratch_long", None), ("ft_trans_long", trans)):
                model = finetune(
                    state, model_cfg, x, lab, (y[lab] - mu) / sd, cfg, 1000 * seed + 7
                )
                pred = predict(model, x, both, starts) * sd + mu
                cal = (y[draw.cal], pred[: len(draw.cal)])
                store.add(
                    arm,
                    "age",
                    args.n,
                    seed,
                    pred[len(draw.cal) :],
                    cal,
                    lr=args.lr,
                    steps=args.steps,
                )
        store.extra["seconds"] = round(time.time() - started, 1)
        store.save(args.out)
        print("saved", fold.name, store.extra["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
