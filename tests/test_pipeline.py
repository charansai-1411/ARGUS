"""Smoke tests on a tiny synthetic file in the IBM AML format (no real data needed)."""

import numpy as np
import pandas as pd

from argus.data.ibm import attach_rings, load_patterns, load_transactions
from argus.data.split import chronological_split
from argus.eval.metrics import best_f1_threshold, evaluate, ring_recall
from argus.features.tabular import FEATURES, build_features
from argus.models.rules import RuleBaseline


def test_load_and_attach_rings(raw_dir):
    tx = load_transactions(raw_dir, "T")
    assert len(tx) == 404 and tx["ts"].is_monotonic_increasing
    assert not tx["src_bank"].str.match(r"0\d").any()  # zero-padding normalised
    tx = attach_rings(tx, load_patterns(raw_dir, "T"))
    assert (tx.ring_id == 0).sum() == 4
    assert (tx.loc[tx.ring_id == 0, "is_laundering"] == 1).all()


def test_split_is_chronological(raw_dir):
    tx = load_transactions(raw_dir, "T")
    s = chronological_split(tx, (0.6, 0.2, 0.2))
    assert tx.ts[s == "train"].max() < tx.ts[s == "val"].min()
    assert tx.ts[s == "val"].max() < tx.ts[s == "test"].min()


def test_features_past_only(raw_dir):
    tx = load_transactions(raw_dir, "T")
    X = build_features(tx)
    assert list(X.columns) == FEATURES and len(X) == len(tx)
    # Features of early rows must not change when later rows are removed.
    X_head = build_features(tx.iloc[:200].copy())
    cols = [c for c in FEATURES if c not in ("payment_format", "pay_currency")]
    pd.testing.assert_frame_equal(X.iloc[:200][cols], X_head[cols])


def test_ring_recall():
    rings = np.array([0, 0, 1, 1, -1])
    assert ring_recall(rings, [1, 1, 1, 0, 0], 1.0) == 0.5
    assert ring_recall(rings, [1, 1, 1, 0, 0], 0.5) == 1.0


def test_rules_and_evaluate(raw_dir):
    tx = attach_rings(load_transactions(raw_dir, "T"), load_patterns(raw_dir, "T"))
    X = build_features(tx)
    scores = RuleBaseline(0.9, 0.9).fit(X).score(X)
    y = tx.is_laundering.to_numpy()
    r = evaluate(y, scores, best_f1_threshold(y, scores), tx.ring_id)
    assert 0 <= r["pr_auc"] <= 1 and "ring_recall@1" in r
