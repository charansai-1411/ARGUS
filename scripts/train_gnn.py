"""Phase 2: train the heterogeneous edge-classification GNN on IBM AML.

Leakage-safe protocol (same as the baselines, plus graph phases):
  train on train transfers using the train-period graph,
  select by val PR-AUC on val transfers using the train+val graph,
  report on test transfers using the full graph, with the val-chosen threshold.

Usage (local quick check):
    python scripts/train_gnn.py --limit 300000 --epochs 1
Kaggle / Colab: see notebooks/train_gnn_kaggle.ipynb
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from argus.config import load_config, resolve
from argus.data.ibm import attach_rings, load_accounts, load_patterns, load_transactions
from argus.eval.metrics import best_f1_threshold, evaluate
from argus.graph.build import GRAPH_VERSION, TRANSFER, build_hetero_graph
from argus.graph import trainer


def get_graph(cfg, raw_dir: Path, processed_dir: Path, limit=None, rebuild=False):
    dc = cfg["data"]
    tag = f"{dc['variant']}" + (f"_limit{limit}" if limit else "")
    cache = processed_dir / f"{tag}_graph.pt"
    if cache.exists() and not rebuild:
        data = torch.load(cache, weights_only=False)
        if getattr(data, "graph_version", 1) == GRAPH_VERSION:
            return data
        print("cached graph is from an older feature version; rebuilding")
    tx = load_transactions(raw_dir, dc["variant"], cache_dir=processed_dir)
    if limit:
        tx = tx.iloc[:limit].copy()
    patterns = raw_dir / f"{dc['variant']}_Patterns.txt"
    tx = attach_rings(tx, load_patterns(raw_dir, dc["variant"])) if patterns.exists() else tx.assign(ring_id=-1)
    data = build_hetero_graph(tx, load_accounts(raw_dir, dc["variant"]), dc["split"],
                              cfg["features"]["velocity_window_hours"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw-dir"), ap.add_argument("--processed-dir"), ap.add_argument("--output-dir")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--conv", choices=["gine", "gat", "sage"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the cached graph")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gc = cfg["gnn"]
    if args.epochs: gc["epochs"] = args.epochs
    if args.conv: gc["conv"] = args.conv
    raw_dir = Path(args.raw_dir) if args.raw_dir else resolve(cfg["data"]["raw_dir"])
    processed_dir = Path(args.processed_dir) if args.processed_dir else resolve(cfg["data"]["processed_dir"])
    out_dir = Path(args.output_dir) if args.output_dir else resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(gc["seed"]); np.random.seed(gc["seed"])
    device = torch.device(args.device)
    t0 = time.time()

    data = get_graph(cfg, raw_dir, processed_dir, args.limit, args.rebuild)
    y = data[TRANSFER].y
    ids = trainer.split_ids(data)
    print(data, f"\ngraph ready [{time.time() - t0:.0f}s]  device={device}")
    for s in ids:
        print(f"  {s:5s} {len(ids[s]):>10,} transfers  {y[ids[s]].mean():.4%} laundering")

    model, best, history = trainer.fit(data, gc, device, log=lambda m: print(m, flush=True))
    torch.save(model.state_dict(), out_dir / f"gnn_{gc['conv']}.pt")
    val_scores = trainer.predict(model, data, "val", ids["val"], gc, device)
    test_scores = trainer.predict(model, data, "test", ids["test"], gc, device)
    thr = best_f1_threshold(y[ids["val"]].numpy(), val_scores.numpy())
    result = evaluate(y[ids["test"]].numpy(), test_scores.numpy(), thr,
                      data[TRANSFER].ring_id[ids["test"]].numpy(), cfg["eval"]["ring_recall_thresholds"])
    result.update(conv=gc["conv"], best_val_pr_auc=float(best), history=history, config=gc)

    tag = f"{cfg['data']['variant']}_{gc['conv']}" + (f"_limit{args.limit}" if args.limit else "")
    (out_dir / f"gnn_{tag}.json").write_text(json.dumps(result, indent=2, default=float))
    np.save(out_dir / f"gnn_{tag}_test_scores.npy", test_scores.numpy())
    print(f"TEST  PR-AUC {result['pr_auc']:.4f}  F1 {result['f1']:.4f}  P {result['precision']:.4f}  "
          f"R {result['recall']:.4f}  ring@1 {result.get('ring_recall@1', float('nan')):.3f}  "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
