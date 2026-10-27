"""The attacker's playbook: mutations a launderer uses to evade a detector.

Each operator rewrites the transfers of the rings it is given (and may add accounts or
cover transactions). Operators follow the camouflage taxonomy of CARE-GNN (feature and
relation camouflage) plus the classic structuring tactics:

  separate_owners  fresh mules get distinct owners (hides the shared mule-herder entity)
  add_hops         every transfer a->b becomes a->x->b through an extra mule at another bank
  split            smurfing: each transfer split into 2-4 smaller ones minutes apart
  delay            stretch the ring over hours instead of minutes (beats velocity windows)
  mimic            payment formats/currencies drawn from normal traffic (feature camouflage)
  camouflage       mules transact normally with legitimate accounts (relation camouflage)
"""

import zlib

import numpy as np
import pandas as pd

from argus.twin.rings import clip_to_period

OPERATORS = ["separate_owners", "add_hops", "split", "delay", "mimic", "camouflage"]


class Attacker:
    def __init__(self, world, rng: np.random.Generator, cfg: dict | None = None, tag: str = ""):
        self.w, self.rng, self.cfg, self.tag = world, rng, cfg or {}, tag
        bg = world.background
        self.formats = bg["payment_format"].astype(str).value_counts(normalize=True)
        self.currencies = bg["pay_currency"].astype(str).value_counts(normalize=True)
        self.amounts = bg["amount_paid"].to_numpy()
        self.active = pd.concat([bg["src"], bg["dst"]]).unique()
        self.n_new = 0

    def _new_mule(self, bank, tag):
        self.n_new += 1
        acct = f"{bank}_XA{self.tag}{tag}{self.n_new:06d}"
        return acct, {"acct": acct, "bank": bank, "entity_id": f"HOP_{acct}", "entity_type": "Individual",
                      "bank_name": f"bank {bank}"}

    def _other_bank(self, bank):
        return self.rng.choice([b for b in self.w.banks if b != bank] or [bank])

    # ---- operators: (ring_tx, accounts) -> (ring_tx, accounts) -------------------------------
    def separate_owners(self, rt, acc):
        acc = acc.copy()
        mine = acc["acct"].isin(pd.concat([rt["src"], rt["dst"]]))
        acc.loc[mine, "entity_id"] = "SOLO_" + acc.loc[mine, "acct"]
        return rt, acc

    def add_hops(self, rt, acc):
        rows, new = [], []
        for r in rt[rt["role"] == "ring"].itertuples(index=False):
            x, a = self._new_mule(self._other_bank(r.src_bank), "H")
            new.append(a)
            d = r._asdict()
            t2 = r.ts + pd.Timedelta(minutes=int(self.rng.integers(1, 30)))
            amt2 = round(r.amount_paid * (1 - self.rng.uniform(0.01, 0.03)), 2)
            rows.append({**d, "dst": x, "dst_bank": a["bank"], "dst_acct": x.split("_", 1)[1]})
            rows.append({**d, "ts": t2, "src": x, "src_bank": a["bank"], "src_acct": x.split("_", 1)[1],
                         "amount_paid": amt2, "amount_received": amt2})
        out = pd.concat([pd.DataFrame(rows), rt[rt["role"] != "ring"]], ignore_index=True)
        return out, pd.concat([acc, pd.DataFrame(new)], ignore_index=True)

    def split(self, rt, acc):
        rows = []
        for r in rt.itertuples(index=False):
            d = r._asdict()
            if r.role != "ring":
                rows.append(d); continue
            k = int(self.rng.integers(2, 5))
            for i, part in enumerate(self.rng.dirichlet(np.ones(k) * 3) * r.amount_paid):
                rows.append({**d, "ts": r.ts + pd.Timedelta(minutes=3 * i + int(self.rng.integers(0, 3))),
                             "amount_paid": round(part, 2), "amount_received": round(part, 2)})
        return pd.DataFrame(rows), acc

    def delay(self, rt, acc):
        rt = rt.copy()
        lo, hi = self.cfg.get("delay_factor", (20, 80))
        for rid, g in rt[rt["role"] == "ring"].groupby("ring_id"):
            t0 = g["ts"].min()
            rt.loc[g.index, "ts"] = t0 + (g["ts"] - t0) * float(self.rng.uniform(lo, hi))
        rt["ts"] = rt["ts"].dt.floor("min")
        return rt, acc

    def mimic(self, rt, acc):
        rt = rt.copy()
        m = rt["role"] == "ring"
        rt.loc[m, "payment_format"] = self.rng.choice(self.formats.index, m.sum(), p=self.formats.values)
        cur = self.rng.choice(self.currencies.index, m.sum(), p=self.currencies.values)
        rt.loc[m, "pay_currency"] = cur
        rt.loc[m, "recv_currency"] = cur
        return rt, acc

    def camouflage(self, rt, acc):
        ring = rt[rt["role"] == "ring"]
        lo, hi = self.cfg.get("camouflage_per_mule", (3, 8))
        rows = []
        for rid, g in ring.groupby("ring_id"):
            victim = g.sort_values("ts")["src"].iloc[0]
            t0, t1 = g["ts"].min() - pd.Timedelta(days=2), g["ts"].max() + pd.Timedelta(days=2)
            period = g["ring_period"].iloc[0]
            for mule in set(g["src"]) | set(g["dst"]):
                if mule == victim:
                    continue
                for _ in range(int(self.rng.integers(lo, hi + 1))):
                    other = str(self.rng.choice(self.active))
                    src, dst = (mule, other) if self.rng.random() < 0.5 else (other, mule)
                    amt = round(float(self.rng.choice(self.amounts)), 2)
                    cur = str(self.rng.choice(self.currencies.index, p=self.currencies.values))
                    rows.append({"ts": t0 + (t1 - t0) * float(self.rng.random()),
                                 "src_bank": src.split("_")[0], "src_acct": src.split("_", 1)[1],
                                 "dst_bank": dst.split("_")[0], "dst_acct": dst.split("_", 1)[1],
                                 "amount_received": amt, "recv_currency": cur, "amount_paid": amt,
                                 "pay_currency": cur,
                                 "payment_format": str(self.rng.choice(self.formats.index, p=self.formats.values)),
                                 "is_laundering": 0, "src": src, "dst": dst, "ring_id": -1,
                                 "typology": None, "role": "camouflage", "ring_period": period})
        cover = pd.DataFrame(rows)
        if len(cover):
            cover["ts"] = cover["ts"].dt.floor("min")
        return pd.concat([rt, cover], ignore_index=True), acc

    # ---- playbook ---------------------------------------------------------------------------
    def apply(self, ring_tx, accounts, ops):
        for op in OPERATORS:            # canonical order: the playbook is a set, not a sequence
            if op in ops:
                ring_tx, accounts = getattr(self, op)(ring_tx, accounts)
        return clip_to_period(self.w, ring_tx), accounts


def mutate(world, ring_tx, accounts, ops, seed: int, cfg: dict | None = None, tag: str = ""):
    """Apply playbook `ops` to the given rings, deterministically for (seed, tag, ops).

    `tag` keeps account ids minted by different calls (e.g. per period) from colliding.
    """
    if isinstance(ops, str):  # a bare string would silently become a set of letters
        raise TypeError("ops must be a list of operator names, not a string")
    unknown = set(ops) - set(OPERATORS)
    if unknown:
        raise ValueError(f"unknown attack operators: {sorted(unknown)}")
    key = zlib.crc32((tag + "|" + ",".join(sorted(ops))).encode())
    return Attacker(world, np.random.default_rng([seed, key]), cfg, tag).apply(ring_tx, accounts, set(ops))
