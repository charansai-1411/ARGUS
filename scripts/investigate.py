"""Phase 4: investigation agent + STR drafting on the digital twin.

    python scripts/investigate.py cases                       # template STRs, LightGBM detector
    python scripts/investigate.py cases --llm                 # Groq narrative (needs GROQ_API_KEY)
    python scripts/investigate.py review outputs/cases/ARGUS-00012345.json --officer "V. Rohini" --decision approve

`cases` opens one case per detected test-period ring (seeded at its highest-scoring transfer),
scores ring reconstruction against the twin's ground truth, and writes example case files + STRs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from argus.agent.investigate import Investigator, reconstruction_quality
from argus.agent.llm import narrative_or_none
from argus.agent.report import render_str
from argus.agent.review import record_review
from argus.config import load_config, resolve
from argus.data.ibm import load_accounts, load_transactions
from argus.twin.detectors import GNNDetector, LGBMDetector
from argus.twin.protocol import Experiment
from argus.twin.rings import inject_rings
from argus.twin.world import make_world


def cmd_cases(args, cfg):
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    tc, variant = cfg["twin"], cfg["data"]["variant"]
    raw, proc = resolve(cfg["data"]["raw_dir"]), resolve(cfg["data"]["processed_dir"])
    world = make_world(load_transactions(raw, variant, cache_dir=proc), load_accounts(raw, variant),
                       tc["n_banks"], tc["end"], cfg["data"]["split"])
    rings, ring_acc = inject_rings(world, tc["n_rings"], tc["seed"], tc["rings"])
    exp = Experiment(world, rings, ring_acc, seed=tc["seed"], attack_cfg=tc["attacks"])
    ops = [args.attack] if args.attack else []
    tx, acc, split = exp.version(ops, ops, ops)   # detector retrained on the attacked twin
    det = (LGBMDetector(cfg) if args.detector == "lightgbm" else GNNDetector(cfg, log=lambda m: None)).fit(tx, acc, split)
    scores = np.nan_to_num(det.score(tx, acc, split), nan=0.0)
    log(f"detector {args.detector} ready, threshold {det.thr:.3f}")

    inv = Investigator(tx, acc, scores, det.thr, cfg["agent"]["window_hours"], cfg["agent"]["max_accounts"])
    test = tx[(split == "test") & (tx["role"] == "ring")]
    quality, cases = [], []
    for rid, g in test.groupby("ring_id"):
        s = scores[g.index]
        if s.max() < det.thr:
            quality.append({"ring_id": int(rid), "detected": False})
            continue
        seed = int(g["tx_id"].iloc[int(np.argmax(s))])
        case = inv.build_case(seed)
        truth = set(g["src"]) | set(g["dst"])
        victim = g.sort_values("ts")["src"].iloc[0]
        q = reconstruction_quality(case, truth, victim)
        quality.append({"ring_id": int(rid), "typology": g["typology"].iloc[0], "detected": True, **q})
        cases.append((q["jaccard"], rid, case))
    qdf = pd.DataFrame(quality)
    det_q = qdf[qdf["detected"]]
    summary = {
        "detector": args.detector, "attack": args.attack, "test_rings": int(len(qdf)),
        "rings_detected": int(qdf["detected"].sum()),
        "mean_precision": round(float(det_q["precision"].mean()), 3),
        "mean_recall": round(float(det_q["recall"].mean()), 3),
        "mean_jaccard": round(float(det_q["jaccard"].mean()), 3),
        "entry_identified": round(float(det_q["entry_correct"].mean()), 3),
        "by_typology": det_q.groupby("typology")[["precision", "recall", "jaccard"]].mean().round(3).to_dict("index"),
    }
    log(f"rings detected {summary['rings_detected']}/{summary['test_rings']}  reconstruction: precision "
        f"{summary['mean_precision']}  recall {summary['mean_recall']}  jaccard {summary['mean_jaccard']}  "
        f"entry found {summary['entry_identified']}")

    out = resolve(cfg["output_dir"]) / "cases"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.detector}" + (f"_{args.attack}" if args.attack else "")
    (resolve(cfg["output_dir"]) / f"agent_eval_{tag}.json").write_text(json.dumps(summary, indent=2))
    # example STRs: one per typology, best-reconstructed first
    picked, seen = [], set()
    for jac, rid, case in sorted(cases, key=lambda c: -c[0]):
        typ = rings.loc[rings["ring_id"] == rid, "typology"].iloc[0]
        if typ not in seen:
            seen.add(typ); picked.append(case)
        if len(picked) >= args.examples:
            break
    for case in picked:
        narrative, meta = narrative_or_none(case, cfg["agent"]["model"], log) if args.llm else (None, None)
        case["narrative_meta"] = meta
        (out / f"{case['case_id']}.json").write_text(json.dumps(case, indent=2, default=str))
        (out / f"{case['case_id']}.md").write_text(render_str(case, narrative), encoding="utf-8")
        log(f"wrote {out / case['case_id']}.md  ({case['indicators']['accounts']} accounts, "
            f"{len(case['freeze_recommendation'])} freeze recommendations)")


def cmd_review(args, cfg):
    case = record_review(Path(args.case), args.officer, args.decision, args.note)
    Path(args.case).with_suffix(".md").write_text(render_str(case), encoding="utf-8")
    print(f"{case['case_id']}: {case['status']} by {case['review']['officer']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--detector", choices=["lightgbm", "gnn"], default="lightgbm")
    c.add_argument("--attack", choices=["split", "mimic", "camouflage", "delay", "add_hops", "separate_owners"])
    c.add_argument("--llm", action="store_true", help="Groq narrative (needs GROQ_API_KEY)")
    c.add_argument("--examples", type=int, default=4)
    r = sub.add_parser("review")
    r.add_argument("case"), r.add_argument("--officer", required=True)
    r.add_argument("--decision", choices=["approve", "reject", "escalate"], required=True)
    r.add_argument("--note", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    {"cases": cmd_cases, "review": cmd_review}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
