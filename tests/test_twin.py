"""Digital twin: ring injection and attack operators keep labels, periods and ids consistent."""

import pandas as pd
import pytest

from argus.data.ibm import load_accounts, load_transactions
from argus.twin.attacks import OPERATORS, mutate
from argus.twin.protocol import Experiment
from argus.twin.rings import TYPOLOGIES, inject_rings
from argus.twin.world import make_world


@pytest.fixture
def world(raw_dir):
    return make_world(load_transactions(raw_dir, "T"), load_accounts(raw_dir, "T"), n_banks=5, end="2022-09-20")


@pytest.fixture
def rings(world):
    return inject_rings(world, 24, seed=1)


def test_world_has_no_original_laundering(world):
    assert world.background["is_laundering"].sum() == 0
    assert world.t_start <= world.t_val <= world.t_test <= world.t_end


def test_injected_rings(world, rings):
    rt, acc = rings
    assert set(rt["typology"]) == set(TYPOLOGIES)
    assert (rt["is_laundering"] == 1).all() and rt["ring_id"].nunique() == 24
    # every transfer of a ring sits in the period the ring started in
    assert (world.period(rt["ts"]) == rt["ring_period"].to_numpy()).all()
    assert acc["acct"].is_unique


@pytest.mark.parametrize("op", OPERATORS)
def test_operator_keeps_invariants(world, rings, op):
    rt, acc = rings
    out, acc2 = mutate(world, rt, acc, [op], seed=3, tag="te")
    ring = out[out["role"] == "ring"]
    assert (ring["is_laundering"] == 1).all()
    assert (out.loc[out["role"] == "camouflage", "is_laundering"] == 0).all()
    assert (world.period(out["ts"]) == out["ring_period"].to_numpy()).all()
    assert acc2["acct"].is_unique
    assert out[["src", "dst", "amount_paid", "payment_format"]].notna().all().all()
    if op == "split":
        assert len(ring) > len(rt)
        assert abs(ring["amount_paid"].sum() - rt["amount_paid"].sum()) < 1.0 * len(ring)
    if op == "add_hops":
        assert len(ring) == 2 * len(rt)
    if op == "camouflage":
        assert (out["role"] == "camouflage").sum() > 0


def test_mutation_is_deterministic(world, rings):
    rt, acc = rings
    a, _ = mutate(world, rt, acc, OPERATORS, seed=5, tag="te")
    b, _ = mutate(world, rt, acc, list(reversed(OPERATORS)), seed=5, tag="te")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_versions_only_touch_their_period(world, rings):
    exp = Experiment(world, *rings, seed=2)
    clean_tx, _, clean_split = exp.version()
    att_tx, att_acc, att_split = exp.version(test=["split", "add_hops"])
    for period in ("train", "val"):
        a = clean_tx[clean_split == period].drop(columns="tx_id").reset_index(drop=True)
        b = att_tx[att_split == period].drop(columns="tx_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
    assert (att_split == "test").sum() > (clean_split == "test").sum()
    assert att_acc["acct"].is_unique
