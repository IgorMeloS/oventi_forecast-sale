import pandas as pd
from typing import Tuple


def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_col: str,
    horizon: int = 4,
    val_size: int = 4,
    test_size: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split based on forecast origins (t).

    We ensure that enough future data exists for multi-horizon targets.

    Returns:
        df_train, df_val, df_test
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    all_dates = df[date_col].sort_values().unique()

    # Last usable origin date (must allow horizon)
    max_origin_idx = len(all_dates) - horizon
    usable_dates = all_dates[:max_origin_idx]

    if len(usable_dates) < (val_size + test_size):
        raise ValueError("Not enough data to perform the requested split.")

    test_dates = usable_dates[-test_size:]
    val_dates = usable_dates[-(test_size + val_size):-test_size]
    train_dates = usable_dates[:-(test_size + val_size)]

    df_train = df[df[date_col].isin(train_dates)].copy()
    df_val = df[df[date_col].isin(val_dates)].copy()
    df_test = df[df[date_col].isin(test_dates)].copy()

    return df_train, df_val, df_test

if __name__ == "__main__":
    from pathlib import Path
    from .io import load_csv

    DATA_PATH = Path("data/ds_assortiment_dataset.csv")

    df = load_csv(DATA_PATH)
    print("Loaded:", df.shape)

    df_train, df_val, df_test = temporal_train_val_test_split(
        df,
        date_col="date",
        horizon=4,
        val_size=4,
        test_size=4,
    )

    print("Train:", df_train.shape)
    print("Val:  ", df_val.shape)
    print("Test: ", df_test.shape)

    print("\nDate ranges:")
    print("Train:", df_train["date"].min(), "→", df_train["date"].max())
    print("Val:  ", df_val["date"].min(), "→", df_val["date"].max())
    print("Test: ", df_test["date"].min(), "→", df_test["date"].max())
