"""LightGBM baseline on tabular features (also the false-positive filter in the full pipeline)."""

import lightgbm as lgb
import numpy as np
import pandas as pd


def train_lgbm(X_tr: pd.DataFrame, y_tr, X_va: pd.DataFrame, y_va, cfg: dict) -> lgb.LGBMClassifier:
    cfg = dict(cfg)
    early = cfg.pop("early_stopping_rounds", 100)
    seed = cfg.pop("seed", 42)
    pos = max(int(np.sum(y_tr)), 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="average_precision",                 # early-stop on PR-AUC, not log-loss
        scale_pos_weight=(len(y_tr) - pos) / pos,   # counter the ~0.1% positive rate
        random_state=seed,
        verbose=-1,
        **cfg,
    )
    model.fit(
        X_tr, y_tr,
        eval_X=(X_va,), eval_y=(y_va,),
        callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(0)],
    )
    return model
