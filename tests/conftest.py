"""Shared fixture: a tiny synthetic dataset in the IBM AML file format."""

import numpy as np
import pandas as pd
import pytest

from argus.data.ibm import RAW_COLUMNS


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2022-09-01")
    rows = []
    for i in range(400):  # normal traffic
        ts = start + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 10)))
        a, b = rng.integers(0, 50, 2)
        amt = round(float(rng.uniform(10, 5000)), 2)
        rows.append([ts.strftime("%Y/%m/%d %H:%M"), f"0{a % 5}", f"800{a:05X}", f"0{b % 5}",
                     f"800{b:05X}", amt, "US Dollar", amt, "US Dollar", "ACH", 0])
    ring = []  # one fan-out ring: hub sends to 4 mules across banks
    for k in range(4):
        ts = (start + pd.Timedelta(days=9, minutes=k)).strftime("%Y/%m/%d %H:%M")
        ring.append([ts, "01", "8AAAAAAAA", f"0{k + 1}", f"8BBBBBB{k:02d}", 9000.0, "Euro", 9000.0,
                     "Euro", "Wire", 1])
    rows += ring
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(tmp_path / "T_Trans.csv", index=False)
    with open(tmp_path / "T_Patterns.txt", "w") as f:
        f.write("BEGIN LAUNDERING ATTEMPT - FAN-OUT\n")
        for r in ring:
            f.write(",".join(map(str, r)) + "\n")
        f.write("END LAUNDERING ATTEMPT - FAN-OUT\n\n")
    accts = sorted({(r[1], r[2]) for r in rows} | {(r[3], r[4]) for r in rows})
    pd.DataFrame(  # every 3 accounts share an owner entity
        [[f"Bank #{b}", b, a, f"E{i // 3:04d}", f"{['Corporation', 'Partnership'][i // 3 % 2]} #{i // 3}"]
         for i, (b, a) in enumerate(accts)],
        columns=["Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name"],
    ).to_csv(tmp_path / "T_accounts.csv", index=False)
    return tmp_path
