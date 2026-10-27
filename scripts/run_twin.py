"""Phase 3: digital twin + adversarial robustness (degradation D, recovery R).

Usage:
    python scripts/run_twin.py --detector lightgbm                # fast, CPU
    python scripts/run_twin.py --detector gnn --epochs 6          # GPU recommended
Writes outputs/twin_<detector>.json
"""

import argparse
import json
import time
from pathlib import Path

from argus.config import load_config, resolve
from argus.data.ibm import load_accounts, load_transactions
from argus.twin.detectors import GNNDetector, LGBMDetector
from argus.twin.protocol import Experiment, run
from argus.twin.rings import inject_rings
from argus.twin.world import make_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--detector", choices=["lightgbm", "gnn"], default="lightgbm")
    ap.add_argument("--raw-dir"), ap.add_argument("--processed-dir"), ap.add_argument("--output-dir")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--n-rings", type=int)
    ap.add_argument("--n-banks", type=int)
    ap.add_argument("--epochs", type=int, help="GNN epochs per fit")
    ap.add_argument("--no-single", action="store_true", help="skip the one-operator-at-a-time sweep")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["twin"]
    if args.rounds is not None: tc["rounds"] = args.rounds
    if args.n_rings: tc["n_rings"] = args.n_rings
    if args.n_banks: tc["n_banks"] = args.n_banks
    if args.epochs: cfg["gnn"]["epochs"] = args.epochs
    raw = Path(args.raw_dir) if args.raw_dir else resolve(cfg["data"]["raw_dir"])
    proc = Path(args.processed_dir) if args.processed_dir else resolve(cfg["data"]["processed_dir"])
    out_dir = Path(args.output_dir) if args.output_dir else resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    variant = cfg["data"]["variant"]
    world = make_world(load_transactions(raw, variant, cache_dir=proc), load_accounts(raw, variant),
                       tc["n_banks"], tc["end"], cfg["data"]["split"])
    rings, ring_acc = inject_rings(world, tc["n_rings"], tc["seed"], tc["rings"])
    log(f"world: {len(world.background):,} background transfers, {len(world.banks)} banks, "
        f"{rings['ring_id'].nunique()} rings / {len(rings):,} laundering transfers "
        f"({rings.groupby('ring_period')['ring_id'].nunique().to_dict()})")

    if args.detector == "lightgbm":
        make = lambda: LGBMDetector(cfg)
    else:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        make = lambda: GNNDetector(cfg, device, log=lambda m: None)
    exp = Experiment(world, rings, ring_acc, seed=tc["seed"], attack_cfg=tc["attacks"])
    result = run(exp, make, rounds=tc["rounds"], single_ops=not args.no_single, log=log)
    result["config"] = {"twin": tc, "detector": args.detector}

    path = out_dir / f"twin_{args.detector}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
