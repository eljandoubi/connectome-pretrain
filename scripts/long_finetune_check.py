"""Does a longer fine-tune change the picture? One held-out site, fixed rates.

usage: python scripts/long_finetune_check.py --data D --ckpt K \
    --out results/long_finetune.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from connpre.data import load_cohort
from connpre.finetune import FinetuneConfig, finetune, predict
from connpre.metrics import mae, r2, spearman
from connpre.model import ModelConfig
from connpre.pretrain import eval_starts
from connpre.protocol import draw_sets, loso_folds
from connpre.settings import MIN_SITE, N_CAL


def scores(y, p) -> dict:
    return {
        "r2": float(r2(y, p)),
        "mae": float(mae(y, p)),
        "spearman": float(spearman(y, p)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", default="NYU")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--steps", default="300,1500")
    ap.add_argument("--lrs", default="1e-4,3e-4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cohort = load_cohort(args.data)
    x = torch.as_tensor(cohort.x, device="cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig(n_roi=x.shape[2])
    fold = next(f for f in loso_folds(cohort.site, MIN_SITE) if f.name == args.site)
    draw = draw_sets(fold.train, cohort.site, N_CAL, args.seed, fold.name)
    lab = draw.labelled(args.n)
    y = cohort.age
    mu, sd = y[lab].mean(), y[lab].std()
    starts = eval_starts(x.shape[1], model_cfg.window, 4)
    trans = torch.load(Path(args.ckpt) / "trans.pt", map_location="cpu")
    out = {"site": fold.name, "n": len(lab), "target": "age", "runs": []}
    for arm, state in (("scratch", None), ("transductive_pretrained", trans)):
        for lr in (float(v) for v in args.lrs.split(",")):
            for steps in (int(v) for v in args.steps.split(",")):
                started = time.time()
                cfg = FinetuneConfig(lr=lr, steps=steps)
                model = finetune(state, model_cfg, x, lab, (y[lab] - mu) / sd, cfg, 7)
                fit = predict(model, x, lab, starts) * sd + mu
                test = predict(model, x, fold.test, starts) * sd + mu
                row = {"arm": arm, "lr": lr, "steps": steps}
                row["train"] = scores(y[lab], fit)
                row["held_out"] = scores(y[fold.test], test)
                row["seconds"] = round(time.time() - started, 1)
                out["runs"].append(row)
                print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
