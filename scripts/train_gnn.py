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
from sklearn.metrics import average_precision_score

from argus.config import load_config, resolve
from argus.data.ibm import attach_rings, load_accounts, load_patterns, load_transactions
from argus.eval.metrics import best_f1_threshold, evaluate
from argus.graph.build import SPLITS, TRANSFER, build_hetero_graph, phase_graph
from argus.graph.model import EdgeGNN
from argus.graph.sampling import make_loader


def get_graph(cfg, raw_dir: Path, processed_dir: Path, limit=None, rebuild=False):
    dc = cfg["data"]
    tag = f"{dc['variant']}" + (f"_limit{limit}" if limit else "")
    cache = processed_dir / f"{tag}_graph.pt"
    if cache.exists() and not rebuild:
        data = torch.load(cache, weights_only=False)
        if "time" in data[TRANSFER]:
            return data
        print("cached graph has no edge timestamps; rebuilding")
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


def step_inputs(batch, data, device):
    ids = batch[TRANSFER].edge_label
    edge_attr = {et: batch[et].edge_attr.to(device) for et in batch.edge_types if "edge_attr" in batch[et]}
    return (
        {nt: batch[nt].x.to(device) for nt in batch.node_types},
        {et: batch[et].edge_index.to(device) for et in batch.edge_types},
        edge_attr,
        batch[TRANSFER].edge_label_index.to(device),
        data[TRANSFER].edge_attr[ids].to(device),
        ids,
    )


@torch.no_grad()
def predict(model, loader, data, device, n_edges):
    model.eval()
    scores = torch.empty(n_edges)
    for batch in loader:
        *inp, ids = step_inputs(batch, data, device)
        scores[ids] = torch.sigmoid(model(*inp)).cpu()
    return scores


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
    split, y = data[TRANSFER].split, data[TRANSFER].y
    ids = {s: torch.nonzero(split == v).squeeze(1) for s, v in SPLITS.items()}
    print(data, f"\ngraph ready [{time.time() - t0:.0f}s]  device={device}")
    for s in ids:
        print(f"  {s:5s} {len(ids[s]):>10,} transfers  {y[ids[s]].mean():.4%} laundering")

    graphs = {s: phase_graph(data, s) for s in SPLITS}
    train_loader = make_loader(graphs["train"], ids["train"], gc, True, gc["batch_size"])
    val_loader = make_loader(graphs["val"], ids["val"], gc, False, gc["eval_batch_size"])
    test_loader = make_loader(graphs["test"], ids["test"], gc, False, gc["eval_batch_size"])

    model = EdgeGNN(data.metadata(), {nt: data[nt].x.size(1) for nt in data.node_types},
                    data[TRANSFER].edge_attr.size(1), gc["hidden"], gc["layers"], gc["conv"],
                    gc["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=gc["lr"], weight_decay=gc["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(gc["pos_weight"], device=device))
    n_edges = data[TRANSFER].num_edges
    ckpt = out_dir / f"gnn_{gc['conv']}.pt"
    best, bad, history = -1.0, 0, []

    for epoch in range(1, gc["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for step, batch in enumerate(train_loader):
            *inp, b_ids = step_inputs(batch, data, device)
            loss = loss_fn(model(*inp), y[b_ids].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b_ids); n += len(b_ids)
            if step % 100 == 0:
                print(f"    epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        val_scores = predict(model, val_loader, data, device, n_edges)[ids["val"]]
        val_ap = average_precision_score(y[ids["val"]], val_scores)
        history.append({"epoch": epoch, "train_loss": total / n, "val_pr_auc": float(val_ap)})
        print(f"epoch {epoch}: loss {total / n:.4f}  val PR-AUC {val_ap:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        if val_ap > best:
            best, bad = val_ap, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= gc["patience"]:
                print("early stop"); break

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    all_val = predict(model, val_loader, data, device, n_edges)
    all_test = predict(model, test_loader, data, device, n_edges)
    thr = best_f1_threshold(y[ids["val"]].numpy(), all_val[ids["val"]].numpy())
    result = evaluate(y[ids["test"]].numpy(), all_test[ids["test"]].numpy(), thr,
                      data[TRANSFER].ring_id[ids["test"]].numpy(), cfg["eval"]["ring_recall_thresholds"])
    result.update(conv=gc["conv"], best_val_pr_auc=float(best), history=history, config=gc)

    tag = f"{cfg['data']['variant']}_{gc['conv']}" + (f"_limit{args.limit}" if args.limit else "")
    (out_dir / f"gnn_{tag}.json").write_text(json.dumps(result, indent=2, default=float))
    np.save(out_dir / f"gnn_{tag}_test_scores.npy", all_test[ids["test"]].numpy())
    print(f"TEST  PR-AUC {result['pr_auc']:.4f}  F1 {result['f1']:.4f}  P {result['precision']:.4f}  "
          f"R {result['recall']:.4f}  ring@1 {result.get('ring_recall@1', float('nan')):.3f}  "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
