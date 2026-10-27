"""Train and score the edge-level GNN (used by scripts/train_gnn.py and the digital twin)."""

import copy
import time

import torch
from sklearn.metrics import average_precision_score

from argus.graph.build import SPLITS, TRANSFER, phase_graph
from argus.graph.model import EdgeGNN
from argus.graph.sampling import make_loader


def split_ids(data) -> dict:
    split = data[TRANSFER].split
    return {s: torch.nonzero(split == v).squeeze(1) for s, v in SPLITS.items()}


def _step_inputs(batch, data, device):
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
def predict(model, data, phase: str, ids, gc: dict, device) -> torch.Tensor:
    """Scores (probabilities) for transfers `ids`, using the graph visible at `phase`."""
    model.eval()
    loader = make_loader(phase_graph(data, phase), ids, gc, False, gc["eval_batch_size"])
    scores = torch.full((data[TRANSFER].num_edges,), float("nan"))
    for batch in loader:
        *inp, b_ids = _step_inputs(batch, data, device)
        scores[b_ids] = torch.sigmoid(model(*inp)).cpu()
    return scores[ids]


def fit(data, gc: dict, device, log=print):
    """Train on train transfers, early-stop on val PR-AUC. Returns (model, best_val_ap, history)."""
    torch.manual_seed(gc["seed"])
    ids = split_ids(data)
    y = data[TRANSFER].y
    t0 = time.time()
    train_loader = make_loader(phase_graph(data, "train"), ids["train"], gc, True, gc["batch_size"])
    model = EdgeGNN(data.metadata(), {nt: data[nt].x.size(1) for nt in data.node_types},
                    data[TRANSFER].edge_attr.size(1), gc["hidden"], gc["layers"], gc["conv"],
                    gc["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=gc["lr"], weight_decay=gc["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(gc["pos_weight"], device=device))
    best, best_state, bad, history = -1.0, None, 0, []

    for epoch in range(1, gc["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for step, batch in enumerate(train_loader):
            *inp, b_ids = _step_inputs(batch, data, device)
            loss = loss_fn(model(*inp), y[b_ids].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b_ids); n += len(b_ids)
            if step % 100 == 0:
                log(f"    epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        val_ap = average_precision_score(y[ids["val"]], predict(model, data, "val", ids["val"], gc, device))
        history.append({"epoch": epoch, "train_loss": total / n, "val_pr_auc": float(val_ap)})
        log(f"epoch {epoch}: loss {total / n:.4f}  val PR-AUC {val_ap:.4f}  [{time.time() - t0:.0f}s]")
        if val_ap > best:
            best, bad, best_state = val_ap, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= gc["patience"]:
                log("early stop")
                break
    model.load_state_dict(best_state)
    return model, float(best), history
