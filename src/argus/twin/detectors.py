"""Detectors with one interface, so the twin can attack and retrain either of them.

fit(tx, accounts, split)   train on 'train' rows, early-stop and pick the F1 threshold on 'val'
score(tx, accounts, split) probability per row ('val' and 'test' rows; NaN for 'train')
"""

import numpy as np

from argus.eval.metrics import best_f1_threshold
from argus.features.tabular import build_features
from argus.models.lgbm import train_lgbm


class LGBMDetector:
    name = "lightgbm"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _X(self, tx):
        return build_features(tx, self.cfg["features"]["velocity_window_hours"])

    def fit(self, tx, accounts, split):
        X, y = self._X(tx), tx["is_laundering"].to_numpy()
        tr, va = split == "train", split == "val"
        self.model = train_lgbm(X[tr], y[tr], X[va], y[va], self.cfg["lightgbm"])
        self.thr = best_f1_threshold(y[va], self.model.predict_proba(X[va])[:, 1])
        return self

    def score(self, tx, accounts, split):
        X = self._X(tx)
        out = np.full(len(tx), np.nan)
        m = split != "train"
        out[m] = self.model.predict_proba(X[m])[:, 1]
        return out


class GNNDetector:
    name = "gnn"

    def __init__(self, cfg: dict, device="cpu", log=print):
        self.cfg, self.device, self.log = cfg, device, log

    def _data(self, tx, accounts, split):
        from argus.graph.build import build_hetero_graph
        return build_hetero_graph(tx, accounts, window_hours=self.cfg["features"]["velocity_window_hours"],
                                  split=split)

    def fit(self, tx, accounts, split):
        from argus.graph import trainer
        data = self._data(tx, accounts, split)
        self.model, self.best_val, _ = trainer.fit(data, self.cfg["gnn"], self.device, log=self.log)
        ids = trainer.split_ids(data)
        val = trainer.predict(self.model, data, "val", ids["val"], self.cfg["gnn"], self.device)
        self.thr = best_f1_threshold(data["account", "transfers", "account"].y[ids["val"]].numpy(), val.numpy())
        return self

    def score(self, tx, accounts, split):
        from argus.graph import trainer
        data = self._data(tx, accounts, split)
        ids = trainer.split_ids(data)
        out = np.full(len(tx), np.nan)
        for phase in ("val", "test"):
            out[ids[phase].numpy()] = trainer.predict(self.model, data, phase, ids[phase],
                                                      self.cfg["gnn"], self.device).numpy()
        return out
