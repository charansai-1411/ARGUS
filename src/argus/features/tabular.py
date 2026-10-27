"""Per-transaction tabular features for the non-graph baselines.

Every feature uses only the transaction itself and the past (events at or before
its timestamp), so the same features are valid at train and test time.
"""

import numpy as np
import pandas as pd

from argus.data.ibm import CURRENCIES, PAYMENT_FORMATS

FEATURES = [
    "log_amount", "amount_ratio", "currency_mismatch", "cross_bank", "self_loop",
    "hour", "weekday", "payment_format", "pay_currency",
    "src_out_win", "src_in_win", "dst_in_win", "dst_out_win",
    "src_out_prior", "dst_in_prior", "src_secs_since_prev_out", "src_amount_vs_mean",
]
CATEGORICAL = ["payment_format", "pay_currency"]


def _window_count(ev_ent, ev_t, q_ent, q_t, window, base):
    """For each query (entity, t), count events of that entity with time in (t - window, t]."""
    keys = np.sort(ev_ent * base + ev_t)
    q = q_ent * base + q_t
    return (np.searchsorted(keys, q, side="right")
            - np.searchsorted(keys, q - window, side="right")).astype("int32")


def build_features(df: pd.DataFrame, window_hours: float = 24) -> pd.DataFrame:
    """Return a feature frame aligned with df (df must be sorted by ts)."""
    f = pd.DataFrame(index=df.index)
    f["log_amount"] = np.log1p(df["amount_paid"])
    f["amount_ratio"] = df["amount_received"] / df["amount_paid"].replace(0, np.nan)
    f["currency_mismatch"] = (df["recv_currency"].astype(str) != df["pay_currency"].astype(str)).astype("int8")
    f["cross_bank"] = (df["src_bank"] != df["dst_bank"]).astype("int8")
    f["self_loop"] = (df["src"] == df["dst"]).astype("int8")
    f["hour"] = df["ts"].dt.hour.astype("int8")
    f["weekday"] = df["ts"].dt.day_of_week.astype("int8")
    f["payment_format"] = pd.Categorical(df["payment_format"].astype(str), categories=PAYMENT_FORMATS)
    f["pay_currency"] = pd.Categorical(df["pay_currency"].astype(str), categories=CURRENCIES)

    # Shared entity codes so an account is the same id whether it sends or receives.
    codes, _ = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True))
    src_c = codes[: len(df)].astype("int64")
    dst_c = codes[len(df):].astype("int64")
    t = ((df["ts"] - df["ts"].min()).dt.total_seconds()).to_numpy().astype("int64")
    w = int(window_hours * 3600)
    base = int(t.max()) + w + 1

    f["src_out_win"] = _window_count(src_c, t, src_c, t, w, base)  # sender velocity
    f["src_in_win"] = _window_count(dst_c, t, src_c, t, w, base)   # sender just received -> pass-through
    f["dst_in_win"] = _window_count(dst_c, t, dst_c, t, w, base)   # receiver fan-in
    f["dst_out_win"] = _window_count(src_c, t, dst_c, t, w, base)  # receiver forwarding

    f["src_out_prior"] = df.groupby("src", sort=False).cumcount().astype("int32")
    f["dst_in_prior"] = df.groupby("dst", sort=False).cumcount().astype("int32")
    prev = df.groupby("src", sort=False)["ts"].shift()
    f["src_secs_since_prev_out"] = (df["ts"] - prev).dt.total_seconds()
    prior_sum = df.groupby("src", sort=False)["amount_paid"].cumsum() - df["amount_paid"]
    prior_mean = prior_sum / f["src_out_prior"].replace(0, np.nan)
    f["src_amount_vs_mean"] = df["amount_paid"] / prior_mean.replace(0, np.nan)
    return f[FEATURES]
