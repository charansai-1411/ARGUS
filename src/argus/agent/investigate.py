"""Deterministic investigation tools: everything in a case file is computed here, never by the LLM.

Starting from one flagged transfer, the investigator
  1. rebuilds the ring: accounts linked by suspicious transfers (score >= threshold) in a time window,
  2. finds the entry (money in from outside the ring, usually the victim) and the exits (cash-out),
  3. traces the trail hop by hop, profiles every account, and lists red-flag indicators,
  4. recommends accounts to freeze — a recommendation for a human officer, never an action.
"""

from collections import deque
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd


class Investigator:
    def __init__(self, tx: pd.DataFrame, accounts: pd.DataFrame, scores, threshold: float,
                 window_hours: float = 48, max_accounts: int = 80):
        self.tx = tx.assign(score=np.asarray(scores, dtype=float)).set_index("tx_id", drop=False)
        self.acc = accounts.drop_duplicates("acct").set_index("acct")
        self.thr, self.window = threshold, pd.Timedelta(hours=window_hours)
        self.max_accounts = max_accounts
        self.sus = self.tx[self.tx["score"] >= threshold]

    # ---- ring reconstruction ------------------------------------------------------------
    def reconstruct(self, seed_tx_id: int) -> pd.DataFrame:
        """Suspicious transfers connected to the seed transfer (BFS over accounts, time-bounded)."""
        seed = self.tx.loc[seed_tx_id]
        win = self.sus[(self.sus["ts"] >= seed["ts"] - self.window) & (self.sus["ts"] <= seed["ts"] + self.window)]
        by_acct = {}
        for tid, s, d in zip(win["tx_id"], win["src"], win["dst"]):
            by_acct.setdefault(s, []).append(tid)
            by_acct.setdefault(d, []).append(tid)
        seen_acct, seen_tx = {seed["src"], seed["dst"]}, {seed_tx_id}
        queue = deque([seed["src"], seed["dst"]])
        while queue and len(seen_acct) < self.max_accounts:
            for tid in by_acct.get(queue.popleft(), []):
                if tid in seen_tx:
                    continue
                seen_tx.add(tid)
                for a in (win.at[tid, "src"], win.at[tid, "dst"]):
                    if a not in seen_acct:
                        seen_acct.add(a)
                        queue.append(a)
        return self.tx.loc[sorted(seen_tx)].sort_values("ts")

    @staticmethod
    def roles(ring: pd.DataFrame) -> dict:
        senders, receivers = set(ring["src"]), set(ring["dst"])
        entry = sorted(senders - receivers) or [ring.sort_values("ts")["src"].iloc[0]]
        exits = sorted(receivers - senders)
        return {a: ("entry" if a in entry else "exit" if a in exits else "mule") for a in senders | receivers}

    # ---- tools ----------------------------------------------------------------------------
    def profile(self, acct: str, ring: pd.DataFrame) -> dict:
        info = self.acc.loc[acct] if acct in self.acc.index else None
        t0 = ring["ts"].min()
        hist = self.tx[(self.tx["src"] == acct) | (self.tx["dst"] == acct)]
        prior = hist[hist["ts"] < t0]
        entity = None if info is None else str(info["entity_id"])
        siblings = [] if entity is None else sorted(set(self.acc.index[self.acc["entity_id"] == entity]) - {acct})
        mine = ring[(ring["src"] == acct) | (ring["dst"] == acct)]
        return {
            "account": acct,
            "bank": acct.split("_")[0],
            "owner_entity": entity,
            "owner_type": None if info is None else str(info["entity_type"]),
            "other_accounts_of_owner": siblings[:10],
            "transfers_before_ring": int(len(prior)),
            "first_seen": None if hist.empty else str(hist["ts"].min()),
            "dormant_before_ring": bool(len(prior) <= 3),
            "received_in_ring": round(float(ring.loc[ring["dst"] == acct, "amount_paid"].sum()), 2),
            "sent_in_ring": round(float(ring.loc[ring["src"] == acct, "amount_paid"].sum()), 2),
            "max_suspicion_score": round(float(mine["score"].max()), 3),
        }

    @staticmethod
    def timeline(ring: pd.DataFrame) -> list:
        t0 = ring["ts"].min()
        return [{"time": str(r.ts), "minute": int((r.ts - t0).total_seconds() // 60), "from": r.src, "to": r.dst,
                 "from_bank": r.src_bank, "to_bank": r.dst_bank, "amount": round(float(r.amount_paid), 2),
                 "currency": str(r.pay_currency), "format": str(r.payment_format), "score": round(float(r.score), 3)}
                for r in ring.sort_values("ts").itertuples()]

    @staticmethod
    def trace(ring: pd.DataFrame, acct: str, direction: str = "down", max_hops: int = 12) -> list:
        """Follow the money from `acct`: 'down' = where it went next, 'up' = where it came from."""
        path, cur, t, seen = [acct], acct, None, {acct}
        for _ in range(max_hops):
            if direction == "down":
                nxt = ring[(ring["src"] == cur) & ((ring["ts"] >= t) if t is not None else True)]
                nxt = nxt.sort_values("amount_paid", ascending=False)
                col = "dst"
            else:
                nxt = ring[(ring["dst"] == cur) & ((ring["ts"] <= t) if t is not None else True)]
                nxt = nxt.sort_values("amount_paid", ascending=False)
                col = "src"
            nxt = nxt[~nxt[col].isin(seen)]
            if nxt.empty:
                break
            row = nxt.iloc[0]
            cur, t = row[col], row["ts"]
            seen.add(cur)
            path.append(cur)
        return path if direction == "down" else path[::-1]

    def indicators(self, ring: pd.DataFrame, profiles: list) -> dict:
        roles = self.roles(ring)
        mules = [a for a, r in roles.items() if r == "mule"]
        pass_through = []
        for m in mules:
            t_in = ring.loc[ring["dst"] == m, "ts"].min()
            t_out = ring.loc[(ring["src"] == m) & (ring["ts"] >= t_in), "ts"].min()
            if pd.notna(t_in) and pd.notna(t_out):
                pass_through.append((t_out - t_in).total_seconds() / 60)
        owners = pd.Series({p["account"]: p["owner_entity"] for p in profiles}).dropna()
        shared = owners[owners.duplicated(keep=False)]
        g = nx.DiGraph(list(zip(ring["src"], ring["dst"])))
        entry = [a for a, r in roles.items() if r == "entry"]
        exits = [a for a, r in roles.items() if r == "exit"]
        return {
            "accounts": len(roles),
            "banks": int(pd.concat([ring["src_bank"], ring["dst_bank"]]).nunique()),
            "transfers": int(len(ring)),
            "cross_bank_share": round(float((ring["src_bank"] != ring["dst_bank"]).mean()), 3),
            "amount_entered": round(float(ring.loc[ring["src"].isin(entry), "amount_paid"].sum()), 2),
            "amount_to_exits": round(float(ring.loc[ring["dst"].isin(exits), "amount_paid"].sum()), 2),
            "duration_minutes": int((ring["ts"].max() - ring["ts"].min()).total_seconds() // 60),
            "median_pass_through_minutes": None if not pass_through else round(float(np.median(pass_through)), 1),
            "dormant_accounts": int(sum(p["dormant_before_ring"] for p in profiles if roles[p["account"]] != "entry")),
            "shared_owner_groups": {str(k): sorted(v.index) for k, v in shared.groupby(shared)} if len(shared) else {},
            "max_fan_out": int(max(dict(g.out_degree()).values())),
            "max_fan_in": int(max(dict(g.in_degree()).values())),
            "has_cycle": bool(next(nx.simple_cycles(g), None) is not None),
            "payment_formats": ring["payment_format"].astype(str).value_counts().to_dict(),
        }

    # ---- case file --------------------------------------------------------------------------
    def build_case(self, seed_tx_id: int, case_id: str | None = None) -> dict:
        ring = self.reconstruct(seed_tx_id)
        roles = self.roles(ring)
        profiles = [{**self.profile(a, ring), "role": roles[a]} for a in sorted(roles, key=lambda a: (roles[a] != "entry", a))]
        entry = [a for a, r in roles.items() if r == "entry"]
        exits = [a for a, r in roles.items() if r == "exit"]
        freeze = [{"account": p["account"], "bank": p["bank"], "role": p["role"], "score": p["max_suspicion_score"],
                   "reason": f"{p['role']} in ring; moved {p['sent_in_ring'] or p['received_in_ring']:,.2f}"
                             + ("; dormant before ring" if p["dormant_before_ring"] else "")}
                  for p in profiles if p["role"] != "entry" and p["max_suspicion_score"] >= self.thr]
        return {
            "case_id": case_id or f"ARGUS-{int(seed_tx_id):08d}",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed_transfer": int(seed_tx_id),
            "status": "pending_review",
            "entry_accounts": entry,
            "exit_accounts": exits,
            "money_trail": self.trace(ring, entry[0], "down") if entry else [],
            "indicators": self.indicators(ring, profiles),
            "accounts": profiles,
            "timeline": self.timeline(ring),
            "freeze_recommendation": sorted(freeze, key=lambda f: -f["score"]),
            "review": None,
        }


def reconstruction_quality(case: dict, true_accounts: set, true_entry: str | None = None) -> dict:
    got = {p["account"] for p in case["accounts"]}
    inter = len(got & true_accounts)
    return {"precision": inter / len(got) if got else 0.0,
            "recall": inter / len(true_accounts) if true_accounts else 0.0,
            "jaccard": inter / len(got | true_accounts) if got | true_accounts else 0.0,
            "entry_correct": None if true_entry is None else true_entry in case["entry_accounts"]}
