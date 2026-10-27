import numpy as np
import pandas as pd


def chronological_split(df: pd.DataFrame, fractions=(0.6, 0.2, 0.2)) -> np.ndarray:
    """Return an array of 'train' / 'val' / 'test' labels split by time, never at random.

    A random split leaks the future into training (the model sees later hops of a
    ring it is asked to detect), which inflates every AML metric.
    Assumes df is sorted by ts; cut points snap to timestamp boundaries so a
    single minute is never split across sets.
    """
    assert abs(sum(fractions) - 1) < 1e-9
    n = len(df)
    ts = df["ts"].to_numpy()
    cut1 = np.searchsorted(ts, ts[int(n * fractions[0])], side="left")
    cut2 = np.searchsorted(ts, ts[int(n * (fractions[0] + fractions[1]))], side="left")
    split = np.empty(n, dtype=object)
    split[:cut1], split[cut1:cut2], split[cut2:] = "train", "val", "test"
    return split
