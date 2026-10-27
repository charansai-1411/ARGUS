"""Phase 5: precompute everything the live demo replays (so the demo also runs offline).

    python scripts/build_demo.py                       # GNN detectors (~6 min CPU)
    python scripts/build_demo.py --detector lightgbm   # faster
Writes api/demo_data.json.

Story: a ring in the test period of the twin.
  Beats 1-3: clean twin, detector trained on clean data catches the ring; the agent opens a case.
  Beat 4:    the same ring under the camouflage attack. The naive detector (trained clean) misses
             most of it; the hardened detector (retrained on attacked data) catches it.
  Beat 5:    headline results from outputs/*.json.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from argus.agent.investigate import Investigator, reconstruction_quality
from argus.agent.llm import narrative_or_none
from argus.agent.report import red_flags, render_str
from argus.config import load_config, resolve
from argus.data.ibm import load_accounts, load_transactions
from argus.twin.detectors import GNNDetector, LGBMDetector
from argus.twin.protocol import Experiment
from argus.twin.rings import inject_rings
from argus.twin.world import make_world

ATTACK = "camouflage"


def ring_graph(tx, scores_by_model: dict, ring_id: int, roles: dict, context: int = 18, seed: int = 0):
    """Nodes/edges for the featured ring plus a few legitimate counterparties for context."""
    ring = tx[(tx["ring_id"] == ring_id) & (tx["role"] == "ring")]
    accts = set(ring["src"]) | set(ring["dst"])
    t0, t1 = ring["ts"].min() - pd.Timedelta(days=1), ring["ts"].max() + pd.Timedelta(days=1)
    cover = tx[(tx["role"] == "camouflage") & (tx["src"].isin(accts) | tx["dst"].isin(accts))
               & (tx["ts"] >= t0 - pd.Timedelta(days=2)) & (tx["ts"] <= t1 + pd.Timedelta(days=2))]
    normal = tx[(tx["role"] == "background") & (tx["src"].isin(accts) | tx["dst"].isin(accts))
                & (tx["ts"] >= t0) & (tx["ts"] <= t1)]
    normal = normal.sample(min(len(normal), context), random_state=seed) if len(normal) else normal
    sub = pd.concat([ring, cover, normal]).drop_duplicates("tx_id").sort_values("ts")
    start = ring["ts"].min()

    edges = []
    for r in sub.itertuples():
        e = {"id": f"t{r.tx_id}", "source": r.src, "target": r.dst, "time": str(r.ts),
             "minute": int((r.ts - start).total_seconds() // 60), "amount": round(float(r.amount_paid), 2),
             "currency": str(r.pay_currency), "format": str(r.payment_format), "kind": r.role}
        for name, s in scores_by_model.items():
            e[f"score_{name}"] = round(float(s[r.Index]), 4)
        edges.append(e)
    nodes = []
    for a in sorted(set(sub["src"]) | set(sub["dst"])):
        role = roles.get(a, "legit")
        mine = [e for e in edges if a in (e["source"], e["target"])]
        n = {"id": a, "bank": a.split("_")[0], "role": role}
        for name in scores_by_model:
            n[f"score_{name}"] = max((e[f"score_{name}"] for e in mine if e["kind"] == "ring"), default=0.0)
        nodes.append(n)
    return {"nodes": nodes, "edges": edges}


def ring_catch(tx, scores, thr, ring_id):
    m = ((tx["ring_id"] == ring_id) & (tx["role"] == "ring")).to_numpy()
    flagged = int((scores[m] >= thr).sum())
    return {"flagged": flagged, "transfers": int(m.sum()), "ring_recall": round(flagged / max(1, m.sum()), 3)}


def load_json(path):
    p = resolve(path)
    return json.loads(p.read_text()) if p.exists() else None


def results_block(cfg):
    base = load_json("outputs/baselines_HI-Small.json") or {}
    rows = [{"model": "Rules", **{k: base.get("rules", {}).get(k) for k in ("pr_auc", "f1", "precision", "recall")}},
            {"model": "LightGBM", **{k: base.get("lightgbm", {}).get(k) for k in ("pr_auc", "f1", "precision", "recall")}},
            # Full HI-Small GNN run happens on Kaggle (2 h, GPU); numbers recorded in the README.
            {"model": "GNN (GINE)", "pr_auc": 0.7258, "f1": 0.7291, "precision": 0.8607, "recall": 0.6324}]
    robust = {}
    for det in ("lightgbm", "gnn"):
        j = load_json(f"outputs/twin_{det}.json")
        if j:
            robust[det] = {"clean_f1": j["clean"]["f1"],
                           "single": {op: {"D": v["D"], "R": v["R"], "attack_f1": v["attack"]["f1"],
                                           "retrained_f1": v["retrained"]["f1"]} for op, v in j.get("single", {}).items()},
                           "adaptive": [{"round": r["round"], "added": r["added"], "before": r["before_f1"],
                                         "attack": r["attack"]["f1"], "retrained": r["retrained"]["f1"]}
                                        for r in j.get("adaptive", [])]}
    agent = {k: load_json(f"outputs/agent_eval_lightgbm{s}.json") for k, s in
             (("clean", ""), ("split", "_split"), ("camouflage", "_camouflage"))}
    return {"hi_small": rows, "robustness": robust, "agent": {k: v for k, v in agent.items() if v}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--detector", choices=["gnn", "lightgbm"], default="gnn")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["gnn"]["epochs"] = args.epochs
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    tc, variant = cfg["twin"], cfg["data"]["variant"]
    raw, proc = resolve(cfg["data"]["raw_dir"]), resolve(cfg["data"]["processed_dir"])
    world = make_world(load_transactions(raw, variant, cache_dir=proc), load_accounts(raw, variant),
                       tc["n_banks"], tc["end"], cfg["data"]["split"])
    rings, ring_acc = inject_rings(world, tc["n_rings"], tc["seed"], tc["rings"])
    exp = Experiment(world, rings, ring_acc, seed=tc["seed"], attack_cfg=tc["attacks"])
    make = (lambda: LGBMDetector(cfg)) if args.detector == "lightgbm" else (lambda: GNNDetector(cfg, log=lambda m: None))

    clean = exp.version()
    attacked_test = exp.version(test=[ATTACK])          # naive model faces this
    attacked_all = exp.version([ATTACK], [ATTACK], [ATTACK])
    naive = make().fit(*clean); log("naive detector trained (clean twin)")
    hard = make().fit(*attacked_all); log(f"hardened detector trained ({ATTACK}-attacked twin)")
    s_clean = np.nan_to_num(naive.score(*clean), nan=0.0)
    s_naive = np.nan_to_num(naive.score(*attacked_test), nan=0.0)
    s_hard = np.nan_to_num(hard.score(*attacked_test), nan=0.0)

    # featured ring: caught on the clean twin, rebuilt exactly by the agent (so the case numbers
    # describe the ring on screen), missed by the naive model under attack, caught by the hardened one
    tx_c, tx_a = clean[0], attacked_test[0]
    inv = Investigator(tx_c, clean[1], s_clean, naive.thr, cfg["agent"]["window_hours"], cfg["agent"]["max_accounts"])
    test_rings = sorted(tx_a.loc[(attacked_test[2] == "test") & (tx_a["role"] == "ring"), "ring_id"].unique())
    best, best_key, cases = None, None, {}
    for rid in test_rings:
        c = ring_catch(tx_c, s_clean, naive.thr, rid)
        n = ring_catch(tx_a, s_naive, naive.thr, rid)
        h = ring_catch(tx_a, s_hard, hard.thr, rid)
        ring_rows = tx_c[(tx_c["ring_id"] == rid) & (tx_c["role"] == "ring")].sort_values("ts")
        truth = set(ring_rows["src"]) | set(ring_rows["dst"])
        jac = 0.0
        if c["ring_recall"] == 1.0:
            cases[rid] = inv.build_case(int(ring_rows["tx_id"].iloc[0]), case_id=f"ARGUS-DEMO-{int(rid):03d}")
            jac = reconstruction_quality(cases[rid], truth, ring_rows["src"].iloc[0])["jaccard"]
        key = (c["ring_recall"] == 1.0, round(jac, 2), h["ring_recall"] - n["ring_recall"], -abs(len(truth) - 8))
        if best_key is None or key > best_key:
            best, best_key = rid, key
    rid = int(best)
    log(f"featured ring {rid} ({rings.loc[rings['ring_id'] == rid, 'typology'].iloc[0]}), "
        f"agent jaccard {best_key[1]}, naive-vs-hardened gap {best_key[2]:.2f}")

    ring_c = tx_c[(tx_c["ring_id"] == rid) & (tx_c["role"] == "ring")].sort_values("ts")
    seed_tx = int(ring_c["tx_id"].iloc[0])                      # the alert: victim's first transfer
    case = cases[rid]
    narrative, meta = (None, None) if args.no_llm else narrative_or_none(case, cfg["agent"]["model"], log)
    case["narrative"], case["narrative_meta"] = narrative, meta
    roles = {p["account"]: p["role"] for p in case["accounts"]}

    seed_row = tx_c.loc[tx_c["tx_id"] == seed_tx].iloc[0]
    demo = {
        "meta": {"detector": args.detector, "attack": ATTACK, "ring_id": rid,
                 "typology": str(ring_c["typology"].iloc[0]), "threshold_naive": round(float(naive.thr), 4),
                 "threshold_hardened": round(float(hard.thr), 4), "banks": [str(b) for b in world.banks],
                 "built_at": time.strftime("%Y-%m-%d %H:%M")},
        "alert": {"tx_id": seed_tx, "from": seed_row["src"], "to": seed_row["dst"], "time": str(seed_row["ts"]),
                  "amount": round(float(seed_row["amount_paid"]), 2), "currency": str(seed_row["pay_currency"]),
                  "format": str(seed_row["payment_format"]),
                  "score": round(float(s_clean[tx_c.index[tx_c["tx_id"] == seed_tx][0]]), 4)},
        "clean": ring_graph(tx_c, {"naive": s_clean}, rid, roles),
        "case": case,
        "red_flags": red_flags(case["indicators"]),
        "str_markdown": render_str(case, narrative),
        "adapt": {**ring_graph(tx_a, {"naive": s_naive, "hardened": s_hard}, rid, roles),
                  "naive": ring_catch(tx_a, s_naive, naive.thr, rid),
                  "hardened": ring_catch(tx_a, s_hard, hard.thr, rid),
                  "clean_naive": ring_catch(tx_c, s_clean, naive.thr, rid)},
        "results": results_block(cfg),
    }
    out = resolve("api/demo_data.json")
    out.write_text(json.dumps(demo, indent=1, default=str), encoding="utf-8")
    log(f"wrote {out}  naive under attack {demo['adapt']['naive']}  hardened {demo['adapt']['hardened']}")


if __name__ == "__main__":
    main()
