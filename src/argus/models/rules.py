"""Rule-based baseline: the fixed-threshold logic banks use today, which ARGUS must beat."""

import numpy as np
import pandas as pd


class RuleBaseline:
    """Score = number of rules fired. Thresholds are quantiles fitted on train only."""

    def __init__(self, amount_quantile: float = 0.99, velocity_quantile: float = 0.99):
        self.aq, self.vq = amount_quantile, velocity_quantile

    def fit(self, X: pd.DataFrame):
        self.amount_thr = X["log_amount"].quantile(self.aq)
        self.src_vel_thr = X["src_out_win"].quantile(self.vq)
        self.dst_vel_thr = X["dst_in_win"].quantile(self.vq)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        rules = [
            X["log_amount"] >= self.amount_thr,                       # large transfer
            X["src_out_win"] >= self.src_vel_thr,                     # sender fan-out burst
            X["dst_in_win"] >= self.dst_vel_thr,                      # receiver fan-in burst
            (X["src_in_win"] > 0) & (X["cross_bank"] == 1),           # money in, then straight out to another bank
            X["currency_mismatch"] == 1,                              # currency hop
        ]
        return np.sum([r.to_numpy() for r in rules], axis=0).astype(float)
