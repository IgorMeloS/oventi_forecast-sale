# src/forecast_sales/features.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig


def _check_and_sort(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    missing = [c for c in (cfg.id_cols + [cfg.date_col, cfg.target_col]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col])
    out = out.sort_values(cfg.id_cols + [cfg.date_col]).reset_index(drop=True)
    return out


def _add_calendar_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if not cfg.use_calendar:
        return df

    out = df.copy()
    ds = out[cfg.date_col]
    month = ds.dt.month.astype(int)
    quarter = ds.dt.quarter.astype(int)

    # cyclical month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["quarter"] = quarter

    # time_idx global (monotonic)
    # If you want per-series time_idx, change this.
    out["time_idx"] = (ds.dt.year - ds.dt.year.min()) * 12 + (ds.dt.month - 1)
    return out


def _add_target_lags(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(cfg.id_cols, sort=False)[cfg.target_col]
    for k in cfg.lags:
        out[f"lag_{k}"] = g.shift(k)
    return out


def _add_target_rollings(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(cfg.id_cols, sort=False)[cfg.target_col]

    # Important: shift(1) first to prevent leakage at time t
    shifted = g.shift(1)

    for w in cfg.rolling_windows:
        if "mean" in cfg.rolling_stats:
            out[f"roll_mean_{w}"] = shifted.groupby(out[cfg.id_cols].apply(tuple, axis=1)).transform(
                lambda s: s.rolling(window=w, min_periods=w).mean()
            )
        if "std" in cfg.rolling_stats:
            out[f"roll_std_{w}"] = shifted.groupby(out[cfg.id_cols].apply(tuple, axis=1)).transform(
                lambda s: s.rolling(window=w, min_periods=w).std(ddof=0)
            )
    return out


def _add_intermittence_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if not cfg.use_intermittence:
        return df

    out = df.copy()
    y = out[cfg.target_col]
    g = out.groupby(cfg.id_cols, sort=False)[cfg.target_col]

    # is_zero_lag1 uses lag_1 if already computed, else compute quickly
    if "lag_1" in out.columns:
        lag1 = out["lag_1"]
    else:
        lag1 = g.shift(1)
        out["lag_1"] = lag1

    out["is_zero_lag1"] = (lag1.fillna(0) == 0).astype(int)

    # zeros_last_k based on shifted values (t-1..)
    shifted = g.shift(1)
    def zeros_count_last_k(s: pd.Series) -> pd.Series:
        return s.rolling(cfg.zeros_window, min_periods=cfg.zeros_window).apply(lambda a: np.sum(a == 0), raw=True)

    out[f"zeros_last_{cfg.zeros_window}"] = shifted.groupby(out[cfg.id_cols].apply(tuple, axis=1)).transform(zeros_count_last_k)

    # nonzero rate on last N months (shifted)
    def nonzero_rate(s: pd.Series) -> pd.Series:
        return s.rolling(cfg.nonzero_window, min_periods=cfg.nonzero_window).apply(lambda a: np.mean(a > 0), raw=True)

    out[f"nonzero_rate_{cfg.nonzero_window}"] = shifted.groupby(out[cfg.id_cols].apply(tuple, axis=1)).transform(nonzero_rate)

    # time since last sale (TSLS): months since last y>0 (using only past)
    # We'll compute from shifted (so at t it doesn't see y_t)
    def tsls(s: pd.Series) -> pd.Series:
        # s is shifted series
        last_pos = np.full(len(s), np.nan)
        last_seen = -10**9
        for i, v in enumerate(s.values):
            if np.isnan(v):
                last_pos[i] = np.nan
            else:
                if v > 0:
                    last_seen = i
                last_pos[i] = i - last_seen if last_seen > -10**8 else np.nan
        # cap
        out_arr = np.minimum(last_pos, cfg.tsls_cap)
        return pd.Series(out_arr, index=s.index)

    out["time_since_last_sale"] = shifted.groupby(out[cfg.id_cols].apply(tuple, axis=1)).transform(tsls)

    return out


def _add_exogenous_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    keys = cfg.id_cols
    g = out.groupby(keys, sort=False)

    # Price features
    if cfg.use_price and cfg.price_col in out.columns:
        price = out[cfg.price_col].astype(float)
        out["price_lag1"] = g[cfg.price_col].shift(1)
        for w in cfg.price_roll_windows:
            shifted = g[cfg.price_col].shift(1)
            out[f"price_roll_mean_{w}"] = shifted.groupby(out[keys].apply(tuple, axis=1)).transform(
                lambda s: s.rolling(window=w, min_periods=w).mean()
            )

    # Discount features
    if cfg.use_discount and cfg.discount_col in out.columns:
        disc = out[cfg.discount_col].astype(float).clip(cfg.discount_clip_min, cfg.discount_clip_max)
        out[cfg.discount_col] = disc  # sanitize in-place (optional; or keep separate)
        out["has_discount_lag1"] = (g[cfg.discount_col].shift(1).fillna(0) > 0).astype(int)
        shifted = g[cfg.discount_col].shift(1)
        for w in cfg.discount_roll_windows:
            out[f"discount_roll_mean_{w}"] = shifted.groupby(out[keys].apply(tuple, axis=1)).transform(
                lambda s: s.rolling(window=w, min_periods=w).mean()
            )

    return out


def make_multi_horizon_targets(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Adds y_h{h} columns using future shifts. Only use in TRAIN mode."""
    out = df.copy()
    g = out.groupby(cfg.id_cols, sort=False)[cfg.target_col]
    for h in cfg.horizons:
        out[f"y_h{h}"] = g.shift(-h)
    return out


def fit_series_profile(df_train: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Train-only series profile (anti-leakage).
    Returns a DataFrame keyed by (store_id, product_id).
    """
    if not cfg.use_series_profile:
        return pd.DataFrame(columns=cfg.id_cols)

    grp = df_train.groupby(cfg.id_cols, sort=False)[cfg.target_col]
    prof = grp.agg(
        series_mean_train="mean",
        series_std_train=lambda s: float(np.std(s.values, ddof=0)),
        series_nonzero_rate_train=lambda s: float(np.mean(s.values > 0)),
    ).reset_index()

    # series_age_train: months since first non-zero in TRAIN (or observed length)
    def age_from_first_nonzero(s: pd.Series) -> float:
        idx = np.where(s.values > 0)[0]
        if len(idx) == 0:
            return float(len(s))
        return float(len(s) - idx[0])

    age = grp.apply(age_from_first_nonzero).reset_index().rename(columns={cfg.target_col: "series_age_train"})
    prof = prof.merge(age, on=cfg.id_cols, how="left")

    return prof


def apply_series_profile(df: pd.DataFrame, series_profile: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if (series_profile is None) or (len(series_profile) == 0) or (not cfg.use_series_profile):
        return df
    out = df.merge(series_profile, on=cfg.id_cols, how="left")
    return out


def build_features(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    mode: Literal["train", "predict"],
    series_profile: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    """
    Returns:
      X: features dataframe including id_cols + date_col
      Y: target dataframe (train only) with columns y_h1..y_h4
      meta: dict with feature_names and categorical features
    """
    df0 = _check_and_sort(df, cfg)

    # Base feature blocks
    feat = df0.copy()
    feat = _add_calendar_features(feat, cfg)
    feat = _add_target_lags(feat, cfg)
    feat = _add_target_rollings(feat, cfg)
    feat = _add_intermittence_features(feat, cfg)
    feat = _add_exogenous_features(feat, cfg)

    # Series profile (train-only fitted elsewhere)
    feat = apply_series_profile(feat, series_profile, cfg)

    Y = None
    if mode == "train":
        feat = make_multi_horizon_targets(feat, cfg)
        y_cols = [f"y_h{h}" for h in cfg.horizons]
        Y = feat[y_cols].copy()

    # Define feature columns (exclude raw target and future targets)
    drop_cols = {cfg.target_col}
    if mode == "train":
        drop_cols |= {f"y_h{h}" for h in cfg.horizons}

    # Keep identifiers + ds in X (useful for joining/debugging)
    base_cols = cfg.id_cols + [cfg.date_col]
    feature_cols = [c for c in feat.columns if c not in drop_cols and c not in base_cols]

    X = feat[base_cols + feature_cols].copy()

    meta = {
        "feature_names": feature_cols,
        "cat_features": cfg.id_cols,
        "config": asdict(cfg),
    }
    return X, Y, meta


if __name__ == "__main__":
    from pathlib import Path

    print("=== FEATURES SMOKE TEST ===")

    # Path à adapter si besoin
    DATA_PATH = Path("data/ds_assortiment_dataset.csv")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH.resolve()}")

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape}")

    # Basic config
    cfg = FeatureConfig()

    # Ensure date column name
    if "date" in df.columns and cfg.date_col not in df.columns:
        df = df.rename(columns={"date": cfg.date_col})

    # Sort & check
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.id_cols + [cfg.date_col])

    # Simple temporal split (no backtesting here)
    max_date = df[cfg.date_col].max()
    train_cutoff = max_date - pd.DateOffset(months=8)

    df_train = df[df[cfg.date_col] <= train_cutoff].copy()
    df_test = df[df[cfg.date_col] > train_cutoff].copy()

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape:  {df_test.shape}")

    # Fit series profile (TRAIN ONLY)
    series_profile = fit_series_profile(df_train, cfg)
    print(f"Series profile shape: {series_profile.shape}")

    # Build features (TRAIN)
    X_train, Y_train, meta = build_features(
        df_train,
        cfg=cfg,
        mode="train",
        series_profile=series_profile,
    )

    print("\n--- TRAIN FEATURES ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"# features: {len(meta['feature_names'])}")

    print("\nFirst 10 feature names:")
    print(meta["feature_names"][:10])

    print("\nNaN diagnostics (top 10):")
    print(X_train.isna().mean().sort_values(ascending=False).head(10))

    print("\nTarget columns:")
    print(Y_train.columns.tolist())

    # Sanity checks
    assert Y_train.shape[1] == len(cfg.horizons), "Wrong number of horizons"
    assert not any(col.startswith("y_") for col in meta["feature_names"]), "Target leakage in features"

    print("\n Smoke test passed successfully")

