"""Inject mule-ring laundering with known ground truth into the twin.

Each ring moves a victim's money through mule accounts across banks to a cash-out account,
in minutes, following one of four typologies:
  chain          victim -> m1 -> m2 -> ... -> mk -> cash-out
  fan_out        victim -> hub -> {m1..mn} -> one cash-out per mule
  scatter_gather victim -> hub -> {m1..mn} -> collector -> cash-out
  cycle          victim -> m1 -> ... -> mk -> m1 (round trip), m1 -> cash-out
Mules are a mix of recruited low-activity real accounts and fresh accounts; fresh mules of a
ring often share one owner entity (the mule herder).
"""

import numpy as np
import pandas as pd

TYPOLOGIES = ["chain", "fan_out", "scatter_gather", "cycle"]
# Payment formats of laundering in the IBM data (ACH-heavy) vs normal traffic.
LAUNDER_FORMATS = {"ACH": 0.87, "Cheque": 0.06, "Credit Card": 0.04, "Cash": 0.02, "Bitcoin": 0.01}


class RingBuilder:
    def __init__(self, world, rng: np.random.Generator, cfg: dict):
        self.w, self.rng, self.cfg = world, rng, cfg
        bg = world.background
        activity = pd.concat([bg["src"], bg["dst"]]).value_counts()
        self.dormant = activity[activity <= cfg.get("dormant_max_tx", 3)].index.to_numpy()
        self.active = activity.index.to_numpy()
        self.bank_of = world.accounts.set_index("acct")["bank"]
        self.new_accounts = []
        self.rows = []
        self.n_new = 0

    # ---- accounts -------------------------------------------------------------------------
    def new_account(self, bank, entity, kind="mule"):
        self.n_new += 1
        acct = f"{bank}_X{kind[0].upper()}{self.n_new:07d}"
        self.new_accounts.append({"acct": acct, "bank": bank, "entity_id": entity,
                                  "entity_type": "Individual", "bank_name": f"bank {bank}"})
        return acct

    def other_bank(self, bank):
        choices = [b for b in self.w.banks if b != bank]
        return self.rng.choice(choices) if choices else bank

    def mules(self, ring_id, n, first_bank):
        herder = f"HERDER_{ring_id}" if self.rng.random() < self.cfg.get("p_shared_owner", 0.6) else None
        out, bank = [], first_bank
        for i in range(n):
            bank = self.other_bank(bank)
            if self.rng.random() < self.cfg.get("p_recruit", 0.5) and len(self.dormant):
                out.append(str(self.rng.choice(self.dormant)))
            else:
                out.append(self.new_account(bank, herder or f"MULE_{ring_id}_{i}"))
        return out

    def bank(self, acct):
        return acct.split("_")[0]

    # ---- transfers ------------------------------------------------------------------------
    def transfer(self, ring_id, typology, src, dst, amount, ts):
        fmt = self.rng.choice(list(LAUNDER_FORMATS), p=list(LAUNDER_FORMATS.values()))
        amount = round(float(amount), 2)
        self.rows.append({"ts": ts, "src_bank": self.bank(src), "src_acct": src.split("_", 1)[1],
                          "dst_bank": self.bank(dst), "dst_acct": dst.split("_", 1)[1],
                          "amount_received": amount, "recv_currency": "US Dollar", "amount_paid": amount,
                          "pay_currency": "US Dollar", "payment_format": fmt, "is_laundering": 1,
                          "src": src, "dst": dst, "ring_id": ring_id, "typology": typology, "role": "ring"})

    def gap(self):
        return pd.Timedelta(minutes=max(1, int(self.rng.exponential(self.cfg.get("hop_gap_min", 15)))))

    def fee(self, amount):
        return amount * (1 - self.rng.uniform(0.01, 0.03))

    # ---- typologies -----------------------------------------------------------------------
    def ring(self, ring_id, typology, start):
        victim = str(self.rng.choice(self.active))
        vbank = self.bank(victim)
        amount = float(np.exp(self.rng.normal(self.cfg.get("log_amount_mean", 9.5), 1.0)))
        t = start
        cash = lambda bank: self.new_account(self.other_bank(bank), f"CASHOUT_{ring_id}_{self.n_new}", "cashout")
        if typology == "chain":
            ms = self.mules(ring_id, int(self.rng.integers(3, 7)), vbank)
            path = [victim, *ms, cash(self.bank(ms[-1]))]
            for a, b in zip(path, path[1:]):
                self.transfer(ring_id, typology, a, b, amount, t); amount, t = self.fee(amount), t + self.gap()
        elif typology in ("fan_out", "scatter_gather"):
            hub, *ms = self.mules(ring_id, int(self.rng.integers(4, 9)), vbank)
            self.transfer(ring_id, typology, victim, hub, amount, t); t += self.gap()
            parts = self.rng.dirichlet(np.ones(len(ms))) * self.fee(amount)
            collector = self.mules(ring_id, 1, self.bank(hub))[0] if typology == "scatter_gather" else None
            for m, part in zip(ms, parts):
                self.transfer(ring_id, typology, hub, m, part, t + self.gap())
            t += pd.Timedelta(minutes=30)
            for m, part in zip(ms, parts):
                dst = collector or cash(self.bank(m))
                self.transfer(ring_id, typology, m, dst, self.fee(part), t + self.gap())
            if collector:
                t += pd.Timedelta(minutes=30)
                self.transfer(ring_id, typology, collector, cash(self.bank(collector)), self.fee(parts.sum() * 0.95), t)
        elif typology == "cycle":
            ms = self.mules(ring_id, int(self.rng.integers(3, 6)), vbank)
            path = [victim, *ms, ms[0]]
            for a, b in zip(path, path[1:]):
                self.transfer(ring_id, typology, a, b, amount, t); amount, t = self.fee(amount), t + self.gap()
            self.transfer(ring_id, typology, ms[0], cash(self.bank(ms[0])), amount, t)
        else:
            raise ValueError(typology)


def inject_rings(world, n_rings: int, seed: int = 0, cfg: dict | None = None):
    """Rings spread uniformly over the twin's time span. Returns (ring_tx, new_accounts)."""
    cfg = cfg or {}
    rng = np.random.default_rng(seed)
    b = RingBuilder(world, rng, cfg)
    span = (world.t_end - world.t_start).total_seconds() - 6 * 3600
    for ring_id in range(n_rings):
        start = world.t_start + pd.Timedelta(seconds=float(rng.uniform(0, span)))
        b.ring(ring_id, TYPOLOGIES[ring_id % len(TYPOLOGIES)], start.floor("min"))
    ring_tx = pd.DataFrame(b.rows)
    ring_tx["ts"] = ring_tx["ts"].clip(upper=world.t_end)
    # A ring belongs to the period its first transfer falls in; mutations keep it there.
    first = ring_tx.groupby("ring_id")["ts"].transform("min")
    ring_tx["ring_period"] = world.period(first)
    return clip_to_period(world, ring_tx), pd.DataFrame(b.new_accounts)


def clip_to_period(world, ring_tx: pd.DataFrame) -> pd.DataFrame:
    """Keep every transfer of a ring inside the period (train/val/test) the ring started in."""
    ring_tx = ring_tx.copy()
    for period in ("train", "val", "test"):
        m = ring_tx["ring_period"] == period
        lo, hi = world.period_bounds(period)
        hi = hi if period == "test" else hi - pd.Timedelta(minutes=1)
        ring_tx.loc[m, "ts"] = ring_tx.loc[m, "ts"].clip(lower=lo, upper=hi)
    return ring_tx
