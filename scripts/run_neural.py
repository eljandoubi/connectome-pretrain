"""Pretrain and run the neural arms; one process handles a shard of the folds.

usage:
  python scripts/run_neural.py --data D --out runs --ckpt ckpt --phase trans
  python scripts/run_neural.py --data D --out runs --ckpt ckpt --mode loso --shard 0/2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from connpre.data import load_cohort
from connpre.finetune import STEPS
from connpre.model import ModelConfig
from connpre.neural import N_MONITOR, fold_neural, pretrain_on
from connpre.pretrain import PretrainConfig
from connpre.protocol import insite_folds, loso_folds
from connpre.settings import INSITE_N, INSITE_SPLITS, MIN_SITE, N_CAL, SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phase", default="folds", choices=["folds", "trans"])
    ap.add_argument("--mode", default="loso", choices=["loso", "insite"])
    ap.add_argument("--folds", default="", help="comma list of fold names")
    ap.add_argument("--shard", default="0/1", help="i/n: handle folds[i::n]")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--sizes", default="50,100,300")
    ap.add_argument("--min-site", type=int, default=MIN_SITE)
    ap.add_argument("--n-cal", type=int, default=N_CAL)
    ap.add_argument("--pre-steps", type=int, default=PretrainConfig.steps)
    ap.add_argument("--scale", type=float, default=1.0, help="scale all step counts")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("gpu", torch.cuda.get_device_name(0), "torch", torch.__version__)
    cohort = load_cohort(args.data)
    x = torch.as_tensor(cohort.x, device=device)
    model_cfg = ModelConfig(n_roi=x.shape[2])
    pre_cfg = PretrainConfig(steps=max(2, int(args.pre_steps * args.scale)))
    out, ckpt = Path(args.out), Path(args.ckpt)
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    if args.phase == "trans":
        started = time.time()
        rng = np.random.default_rng(0)
        monitor = rng.choice(len(cohort), N_MONITOR, replace=False)
        train = np.setdiff1d(np.arange(len(cohort)), monitor)
        args_pre = (pre_cfg, model_cfg, 4242)
        state, history = pretrain_on(x, train, {"val": monitor}, *args_pre)
        torch.save(state, ckpt / "trans.pt")
        (out / "pretrain-trans.json").write_text(json.dumps(history) + "\n")
        print("trans pretrained", round(time.time() - started), "s", history[-1])
        return

    seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.mode == "loso":
        sizes = tuple(int(s) for s in args.sizes.split(","))
        folds = loso_folds(cohort.site, args.min_site)
        trans_path = ckpt / "trans.pt"
        trans = None
        if trans_path.exists():
            trans = torch.load(trans_path, map_location="cpu")
    else:
        sizes = (INSITE_N,)
        folds = insite_folds(cohort.site, INSITE_SPLITS, seed=0)
        trans = None
    keep = set(filter(None, args.folds.split(",")))
    index, count = (int(v) for v in args.shard.split("/"))
    folds = [f for f in folds if not keep or f.name in keep][index::count]
    ft_steps = {n: max(2, int(s * args.scale)) for n, s in STEPS.items()}
    for fold in folds:
        if (out / f"{args.mode}-{fold.name}.json").exists():
            print("skip", args.mode, fold.name)
            continue
        store = fold_neural(
            cohort,
            x,
            fold,
            args.mode,
            seeds,
            sizes,
            args.n_cal,
            pre_cfg,
            model_cfg,
            trans_state=trans,
            ft_steps=ft_steps,
            lr_steps=max(2, int(200 * args.scale)),
            log=lambda msg: print(msg, flush=True),
        )
        store.save(out)
        print("saved", args.mode, fold.name, store.extra["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
