"""Phase 1: load IBM AML, split by time, build features, run rule + LightGBM baselines.

Usage:
    python scripts/run_baselines.py [--config configs/default.yaml] [--limit N]
Writes outputs/baselines_<variant>.json
"""

import argparse
import json
import time

import numpy as np

from argus.config import load_config, resolve
from argus.data.ibm import attach_rings, load_patterns, load_transactions
from argus.data.split import chronological_split
from argus.eval.metrics import best_f1_threshold, evaluate
from argus.features.tabular import build_features
from argus.models.lgbm import train_lgbm
from argus.models.rules import RuleBaseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N transactions (quick runs)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc = cfg["data"]
    t0 = time.time()

    tx = load_transactions(resolve(dc["raw_dir"]), dc["variant"], cache_dir=resolve(dc["processed_dir"]))
    if args.limit:
        tx = tx.iloc[: args.limit].copy()
    patterns_file = resolve(dc["raw_dir"]) / f"{dc['variant']}_Patterns.txt"
    tx = attach_rings(tx, load_patterns(resolve(dc["raw_dir"]), dc["variant"])) if patterns_file.exists() \
        else tx.assign(ring_id=-1)
    print(f"loaded {len(tx):,} transactions, {tx.is_laundering.mean():.4%} laundering, "
          f"{(tx.ring_id >= 0).sum():,} in known rings  [{time.time() - t0:.0f}s]")

    X = build_features(tx, cfg["features"]["velocity_window_hours"])
    y = tx["is_laundering"].to_numpy()
    split = chronological_split(tx, dc["split"])
    m = {s: split == s for s in ("train", "val", "test")}
    for s in m:
        print(f"  {s:5s} {m[s].sum():>10,} tx  {y[m[s]].mean():.4%} positive")
    print(f"features built [{time.time() - t0:.0f}s]")

    results = {}
    ring_thr = cfg["eval"]["ring_recall_thresholds"]

    rules = RuleBaseline(cfg["rules"]["amount_quantile"], cfg["rules"]["velocity_quantile"]).fit(X[m["train"]])
    thr = best_f1_threshold(y[m["val"]], rules.score(X[m["val"]]))
    results["rules"] = evaluate(y[m["test"]], rules.score(X[m["test"]]), thr,
                                tx.loc[m["test"], "ring_id"], ring_thr)

    model = train_lgbm(X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]], cfg["lightgbm"])
    thr = best_f1_threshold(y[m["val"]], model.predict_proba(X[m["val"]])[:, 1])
    results["lightgbm"] = evaluate(y[m["test"]], model.predict_proba(X[m["test"]])[:, 1], thr,
                                   tx.loc[m["test"], "ring_id"], ring_thr)
    results["lightgbm"]["best_iteration"] = int(model.best_iteration_ or 0)
    results["lightgbm"]["feature_importance"] = dict(sorted(
        zip(X.columns, model.booster_.feature_importance("gain").round(1).tolist()),
        key=lambda kv: -kv[1]))

    out = resolve(cfg["output_dir"]) / f"baselines_{dc['variant']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    for name, r in results.items():
        print(f"{name:9s} PR-AUC {r['pr_auc']:.4f}  F1 {r['f1']:.4f}  P {r['precision']:.4f}  "
              f"R {r['recall']:.4f}  ring@1 {r.get('ring_recall@1', np.nan):.3f}")
    print(f"wrote {out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
