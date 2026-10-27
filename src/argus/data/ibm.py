"""Loaders for the IBM AML synthetic dataset (Altman et al., "Realistic Synthetic
Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023 D&B).

Kaggle: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
Expected files in data/raw/: <variant>_Trans.csv and <variant>_Patterns.txt
"""

import re
from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]
RENAME = {
    "Timestamp": "ts", "From Bank": "src_bank", "From Account": "src_acct",
    "To Bank": "dst_bank", "To Account": "dst_acct",
    "Amount Received": "amount_received", "Receiving Currency": "recv_currency",
    "Amount Paid": "amount_paid", "Payment Currency": "pay_currency",
    "Payment Format": "payment_format", "Is Laundering": "is_laundering",
}
# Fixed category lists (from the IBM AML generator) so features keep the same columns
# when the digital twin injects or mutates transactions.
PAYMENT_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
CURRENCIES = ["Australian Dollar", "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro", "Mexican Peso",
              "Ruble", "Rupee", "Saudi Riyal", "Shekel", "Swiss Franc", "UK Pound", "US Dollar", "Yen", "Yuan"]
ENTITY_TYPES = ["Corporation", "Country", "Direct", "Individual", "Partnership", "Sole Proprietorship", "Unknown"]

# Columns that identify one transfer, used to match pattern rows back to transactions.
KEY = ["ts", "src", "dst", "amount_paid", "pay_currency"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    for side in ("src", "dst"):
        # Bank ids appear zero-padded in some files and not in others.
        df[f"{side}_bank"] = df[f"{side}_bank"].astype(str).str.lstrip("0").replace("", "0")
        df[side] = df[f"{side}_bank"] + "_" + df[f"{side}_acct"].astype(str)
    df["amount_paid"] = df["amount_paid"].astype(float).round(2)
    df["amount_received"] = df["amount_received"].astype(float).round(2)
    df["is_laundering"] = df["is_laundering"].astype("int8")
    for col in ("recv_currency", "pay_currency", "payment_format"):
        df[col] = df[col].astype("category")
    return df


def load_transactions(raw_dir: str | Path, variant: str = "HI-Small",
                      cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Load <variant>_Trans.csv as a normalised, time-sorted frame with a stable tx_id."""
    raw_dir = Path(raw_dir)
    cache = Path(cache_dir) / f"{variant}_trans.parquet" if cache_dir else None
    if cache and cache.exists():
        return pd.read_parquet(cache)

    df = pd.read_csv(
        raw_dir / f"{variant}_Trans.csv",
        header=0, names=RAW_COLUMNS,
        dtype={"From Bank": str, "From Account": str, "To Bank": str, "To Account": str},
    )
    df = _normalise(df)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.insert(0, "tx_id", df.index.astype("int64"))

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def load_accounts(raw_dir: str | Path, variant: str = "HI-Small") -> pd.DataFrame:
    """Load <variant>_accounts.csv: one row per account with its owning entity.

    Returns columns: acct (same id as tx src/dst), bank, entity_id, entity_type, bank_name.
    """
    a = pd.read_csv(Path(raw_dir) / f"{variant}_accounts.csv", dtype=str)
    out = pd.DataFrame({
        "bank": a["Bank ID"].str.lstrip("0").replace("", "0"),
        "entity_id": a["Entity ID"],
        "entity_type": a["Entity Name"].str.split(" #").str[0],
        "bank_name": a["Bank Name"],
    })
    out.insert(0, "acct", out["bank"] + "_" + a["Account Number"])
    return out


_BEGIN = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")


def load_patterns(raw_dir: str | Path, variant: str = "HI-Small") -> pd.DataFrame:
    """Parse <variant>_Patterns.txt into one row per transfer with ring_id and typology.

    The file lists each injected laundering attempt between
    'BEGIN LAUNDERING ATTEMPT - <TYPE>' and 'END LAUNDERING ATTEMPT' lines.
    """
    rows, ring_id, typology = [], -1, None
    with open(Path(raw_dir) / f"{variant}_Patterns.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _BEGIN.match(line)
            if m:
                ring_id += 1
                typology = m.group(1).strip("-")
            elif line.startswith("END"):
                typology = None
            elif typology is not None:
                rows.append([*line.split(","), ring_id, typology])
    df = pd.DataFrame(rows, columns=[*RAW_COLUMNS, "ring_id", "typology"])
    return _normalise(df)


def attach_rings(tx: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Add ring_id / typology columns to transactions (-1 / NaN when not in a known ring)."""
    p = patterns[KEY + ["ring_id", "typology"]].drop_duplicates(KEY)
    p["pay_currency"] = p["pay_currency"].astype(str)
    left = tx.assign(pay_currency=tx["pay_currency"].astype(str))
    out = left.merge(p, on=KEY, how="left")
    out["pay_currency"] = out["pay_currency"].astype("category")
    out["ring_id"] = out["ring_id"].fillna(-1).astype("int64")
    return out
