"""Build the heterogeneous transaction graph for the GNN.

Node types : account, entity (owner from <variant>_accounts.csv), bank
Edge types : account -transfers-> account   (one edge per transaction, with features)
             account -rev_transfers-> account (same edges reversed, so messages flow both ways)
             entity -owns-> account, account -owned_by-> entity
             bank -hosts-> account, account -held_at-> bank

The task is edge classification: is this transfer laundering?
Edge features are the past-only tabular features from argus.features.tabular, so the GNN
sees everything LightGBM sees plus the graph around each transfer.
"""

import copy

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from argus.data.split import chronological_split
from argus.features.tabular import CATEGORICAL, build_features

TRANSFER = ("account", "transfers", "account")
REV_TRANSFER = ("account", "rev_transfers", "account")
SPLITS = {"train": 0, "val": 1, "test": 2}
# Heavy-tailed counts / durations / ratios: compress with sign(x)*log1p(|x|) before scaling.
_SLOG = ["amount_ratio", "src_out_win", "src_in_win", "dst_in_win", "dst_out_win",
         "src_out_prior", "dst_in_prior", "src_secs_since_prev_out", "src_amount_vs_mean"]


def _edge_features(tx: pd.DataFrame, train_mask: np.ndarray, window_hours: float):
    f = build_features(tx, window_hours)
    num = f.drop(columns=CATEGORICAL).astype("float64")
    num[_SLOG] = np.sign(num[_SLOG]) * np.log1p(num[_SLOG].abs())
    num = num.replace([np.inf, -np.inf], np.nan)
    missing = num.isna().any().loc[lambda s: s].index
    for c in missing:                                   # keep "no history yet" as its own signal
        num[f"{c}_missing"] = num[c].isna().astype("float64")
    mean, std = num[train_mask].mean(), num[train_mask].std().replace(0, 1)
    num = ((num - mean) / std).fillna(0.0)
    onehot = pd.get_dummies(f[CATEGORICAL].astype(str), dtype="float64")
    feats = pd.concat([num, onehot], axis=1)
    return torch.tensor(feats.to_numpy(), dtype=torch.float32), list(feats.columns)


def build_hetero_graph(tx: pd.DataFrame, accounts: pd.DataFrame, split_fractions=(0.6, 0.2, 0.2),
                       window_hours: float = 24) -> HeteroData:
    """tx must be time-sorted (load_transactions does this) and carry ring_id (attach_rings)."""
    split = np.array([SPLITS[s] for s in chronological_split(tx, split_fractions)], dtype=np.int8)

    acc = accounts.drop_duplicates("acct").set_index("acct")
    tx_accts = pd.unique(pd.concat([tx["src"], tx["dst"]], ignore_index=True))
    acc = acc.reindex(tx_accts)                         # only accounts that transact
    acc["entity_id"] = acc["entity_id"].fillna("unknown_" + acc.index.to_series())
    acc["entity_type"] = acc["entity_type"].fillna("Unknown")
    acc["bank"] = acc["bank"].fillna(acc.index.to_series().str.split("_").str[0])

    acct_idx = pd.Series(np.arange(len(acc)), index=acc.index)
    ent_codes, ent_uniques = pd.factorize(acc["entity_id"])
    bank_codes, bank_uniques = pd.factorize(acc["bank"])

    data = HeteroData()
    types = sorted(acc["entity_type"].unique())
    type_onehot = pd.get_dummies(pd.Categorical(acc["entity_type"], categories=types), dtype="float32").to_numpy()
    data["account"].x = torch.tensor(type_onehot)
    data["account"].num_nodes = len(acc)

    ent_type = pd.Series(acc["entity_type"].to_numpy()).groupby(ent_codes).first()
    ent_onehot = pd.get_dummies(pd.Categorical(ent_type, categories=types), dtype="float32").to_numpy()
    ent_size = np.log1p(np.bincount(ent_codes, minlength=len(ent_uniques))).astype("float32")[:, None]
    data["entity"].x = torch.tensor(np.hstack([ent_onehot, ent_size]))
    data["entity"].num_nodes = len(ent_uniques)

    bank_size = np.log1p(np.bincount(bank_codes, minlength=len(bank_uniques))).astype("float32")[:, None]
    data["bank"].x = torch.tensor(bank_size)
    data["bank"].num_nodes = len(bank_uniques)

    src = torch.tensor(acct_idx.loc[tx["src"]].to_numpy())
    dst = torch.tensor(acct_idx.loc[tx["dst"]].to_numpy())
    edge_attr, feature_names = _edge_features(tx, split == 0, window_hours)
    data[TRANSFER].edge_index = torch.stack([src, dst])
    data[TRANSFER].edge_attr = edge_attr
    data[TRANSFER].y = torch.tensor(tx["is_laundering"].to_numpy(), dtype=torch.float32)
    data[TRANSFER].split = torch.tensor(split)
    data[TRANSFER].ring_id = torch.tensor(tx["ring_id"].to_numpy(), dtype=torch.long)
    data[REV_TRANSFER].edge_index = torch.stack([dst, src])
    data[REV_TRANSFER].edge_attr = edge_attr

    acc_ids = torch.arange(len(acc))
    ent = torch.tensor(ent_codes)
    bank = torch.tensor(bank_codes)
    data["entity", "owns", "account"].edge_index = torch.stack([ent, acc_ids])
    data["account", "owned_by", "entity"].edge_index = torch.stack([acc_ids, ent])
    data["bank", "hosts", "account"].edge_index = torch.stack([bank, acc_ids])
    data["account", "held_at", "bank"].edge_index = torch.stack([acc_ids, bank])

    # Edge timestamps (seconds since the first transfer) for time-respecting sampling:
    # a scored transfer may only see transfers at or before its own time. Ownership and
    # bank links are static, so they get time 0 (always visible).
    t = torch.tensor(((tx["ts"] - tx["ts"].min()).dt.total_seconds()).to_numpy(), dtype=torch.long)
    data[TRANSFER].time = t
    data[REV_TRANSFER].time = t
    for et in data.edge_types:
        if et not in (TRANSFER, REV_TRANSFER):
            data[et].time = torch.zeros(data[et].num_edges, dtype=torch.long)

    data.edge_feature_names = feature_names
    data.entity_types = types
    return data


def phase_graph(data: HeteroData, phase: str) -> HeteroData:
    """Graph visible at a phase, so no future transfers leak into message passing.

    train: train transfers only; val: train + val; test: all transfers.
    Transfers are time-sorted, so each phase is a prefix of the edge list.
    """
    n = int((data[TRANSFER].split <= SPLITS[phase]).sum())
    g = copy.copy(data)  # shallow: tensors are shared, slices below are views (no 5M-edge copies)
    for et in (TRANSFER, REV_TRANSFER):
        for key in ("edge_index", "edge_attr", "y", "split", "ring_id", "time"):
            if key in g[et]:
                t = g[et][key]
                g[et][key] = t[:, :n] if key == "edge_index" else t[:n]
    return g
