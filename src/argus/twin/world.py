"""The multi-bank digital twin: real background traffic + injected rings with known labels.

Background = normal (non-laundering) IBM AML transfers among the `n_banks` busiest banks.
IBM's own laundering rows are dropped, so every positive in the twin is a ring we injected
and its full structure is known (complete ground truth).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TX_COLUMNS = ["ts", "src_bank", "src_acct", "dst_bank", "dst_acct", "amount_received", "recv_currency",
              "amount_paid", "pay_currency", "payment_format", "is_laundering", "src", "dst",
              "ring_id", "typology", "role"]


@dataclass
class World:
    background: pd.DataFrame   # normal transfers, TX_COLUMNS
    accounts: pd.DataFrame     # acct, bank, entity_id, entity_type, bank_name
    banks: list
    t_start: pd.Timestamp
    t_val: pd.Timestamp        # train = [t_start, t_val), val = [t_val, t_test), test = [t_test, t_end]
    t_test: pd.Timestamp
    t_end: pd.Timestamp

    def period(self, ts) -> np.ndarray:
        ts = pd.to_datetime(pd.Series(ts)).to_numpy()
        return np.where(ts < self.t_val.to_datetime64(), "train",
                        np.where(ts < self.t_test.to_datetime64(), "val", "test"))

    def period_bounds(self, period: str):
        return {"train": (self.t_start, self.t_val), "val": (self.t_val, self.t_test),
                "test": (self.t_test, self.t_end)}[period]


def make_world(tx: pd.DataFrame, accounts: pd.DataFrame, n_banks: int = 8, end: str = "2022-09-11",
               split=(0.6, 0.2, 0.2)) -> World:
    """tx: normalised IBM transactions (argus.data.ibm.load_transactions)."""
    vol = pd.concat([tx["src_bank"], tx["dst_bank"]]).value_counts()
    banks = list(vol.index[:n_banks])
    m = (tx["src_bank"].isin(banks) & tx["dst_bank"].isin(banks) & (tx["is_laundering"] == 0)
         & (tx["ts"] < pd.Timestamp(end)))
    bg = tx.loc[m].copy()
    bg["ring_id"], bg["typology"], bg["role"] = -1, None, "background"
    bg = bg[TX_COLUMNS].sort_values("ts", kind="stable").reset_index(drop=True)

    t = bg["ts"].to_numpy()
    t_val = pd.Timestamp(t[int(len(t) * split[0])])
    t_test = pd.Timestamp(t[int(len(t) * (split[0] + split[1]))])
    acc = accounts[accounts["acct"].isin(pd.unique(pd.concat([bg["src"], bg["dst"]])))].copy()
    return World(bg, acc.reset_index(drop=True), banks, pd.Timestamp(t[0]), t_val, t_test, pd.Timestamp(t[-1]))


def assemble(world: World, ring_tx: pd.DataFrame, new_accounts: pd.DataFrame):
    """Merge background + ring transfers into one time-sorted twin. Returns (tx, accounts, split)."""
    tx = pd.concat([world.background, ring_tx[TX_COLUMNS]], ignore_index=True)
    tx = tx.sort_values("ts", kind="stable").reset_index(drop=True)
    tx.insert(0, "tx_id", np.arange(len(tx), dtype="int64"))
    for col in ("recv_currency", "pay_currency", "payment_format"):
        tx[col] = tx[col].astype(str)
    tx["is_laundering"] = tx["is_laundering"].astype("int8")
    tx["ring_id"] = tx["ring_id"].astype("int64")
    accounts = pd.concat([world.accounts, new_accounts], ignore_index=True).drop_duplicates("acct", keep="last")
    return tx, accounts, world.period(tx["ts"])
