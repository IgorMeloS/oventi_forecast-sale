from __future__ import annotations
import pandas as pd
from .config import CFG

def preprocess_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces schema + basic cleaning.
    Expected columns: date, store_id, product_id, sales (names configurable in CFG).
    """
    required = [CFG.date_col, CFG.store_col, CFG.product_col, CFG.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[CFG.date_col] = pd.to_datetime(out[CFG.date_col])

    # Basic sanity
    out[CFG.target_col] = pd.to_numeric(out[CFG.target_col], errors="coerce").fillna(0.0)
    return out
