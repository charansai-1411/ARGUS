"""Attack -> measure the fall -> retrain -> measure the climb.

A playbook is a set of attack operators applied to rings in a period. Notation per round t:
  F1_before  detector's test F1 when test rings match what it was trained on
  F1_attack  same detector, test rings now mutated by the attacker's new playbook
  F1_retrain detector retrained on train/val rings mutated by the same playbook
  D = F1_before - F1_attack      (what the adaptive attacker costs)
  R = F1_retrain - F1_attack     (what retraining wins back)

Adaptive attacker (multi-round, after Prasetya et al. 2025): each round it probes the current
detector on validation rings with every unused operator and adds the one that lowers recall most.
"""

import numpy as np
import pandas as pd

from argus.eval.metrics import evaluate
from argus.twin.attacks import OPERATORS, mutate
from argus.twin.world import assemble

PERIODS = ("train", "val", "test")


class Experiment:
    def __init__(self, world, ring_tx, ring_accounts, seed=0, attack_cfg=None):
        self.w, self.rings, self.ring_acc = world, ring_tx, ring_accounts
        self.seed, self.attack_cfg = seed, attack_cfg or {}

    def version(self, train=(), val=(), test=()):
        """Twin where rings of each period are mutated by that period's playbook."""
        parts, acc = [], self.ring_acc
        for period, ops in zip(PERIODS, (train, val, test)):
            rt = self.rings[self.rings["ring_period"] == period]
            if ops:
                rt, acc = mutate(self.w, rt, acc, ops, self.seed, self.attack_cfg, tag=period[:2])
            parts.append(rt)
        return assemble(self.w, pd.concat(parts, ignore_index=True), acc)


def metrics(det, version, rows="test", ring_thresholds=(1.0, 0.5)):
    tx, acc, split = version
    m = split == rows
    s = det.score(tx, acc, split)[m]
    return evaluate(tx["is_laundering"].to_numpy()[m], s, det.thr, tx["ring_id"].to_numpy()[m], ring_thresholds)


def _short(r):
    keys = ("pr_auc", "f1", "precision", "recall", "ring_recall@1", "ring_recall@0.5")
    return {k: round(float(r[k]), 4) for k in keys if k in r}


def run(exp: Experiment, make_detector, rounds=3, single_ops=True, log=print):
    out = {}
    clean = exp.version()
    det0 = make_detector().fit(*clean)
    base = metrics(det0, clean)
    out["clean"] = _short(base)
    log(f"clean: F1 {base['f1']:.3f}  PR-AUC {base['pr_auc']:.3f}  ring@1 {base['ring_recall@1']:.3f}")

    if single_ops:
        out["single"] = {}
        for op in OPERATORS:
            att = metrics(det0, exp.version(test=[op]))
            v = exp.version([op], [op], [op])
            ret = metrics(make_detector().fit(*v), v)
            out["single"][op] = {"attack": _short(att), "retrained": _short(ret),
                                 "D": round(base["f1"] - att["f1"], 4), "R": round(ret["f1"] - att["f1"], 4)}
            log(f"  {op:15s} F1 attack {att['f1']:.3f}  retrained {ret['f1']:.3f}  "
                f"D {base['f1'] - att['f1']:+.3f}  R {ret['f1'] - att['f1']:+.3f}")

    out["adaptive"] = []
    det, playbook, before = det0, [], base["f1"]
    for t in range(1, rounds + 1):
        unused = [op for op in OPERATORS if op not in playbook]
        if not unused:
            break
        probe = {op: metrics(det, exp.version(playbook, playbook + [op], playbook), rows="val")["recall"]
                 for op in unused}
        chosen = min(probe, key=probe.get)
        new = playbook + [chosen]
        att = metrics(det, exp.version(playbook, playbook, new))
        v = exp.version(new, new, new)
        det = make_detector().fit(*v)
        ret = metrics(det, v)
        rec = {"round": t, "added": chosen, "playbook": new, "probe_val_recall": probe,
               "before_f1": round(before, 4), "attack": _short(att), "retrained": _short(ret),
               "D": round(before - att["f1"], 4), "R": round(ret["f1"] - att["f1"], 4)}
        out["adaptive"].append(rec)
        log(f"round {t}: +{chosen:15s} F1 before {before:.3f} -> attack {att['f1']:.3f} -> retrained "
            f"{ret['f1']:.3f}   D {rec['D']:+.3f}  R {rec['R']:+.3f}")
        playbook, before = new, ret["f1"]
    return out
