from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import FeatureConfig
from .io import load_csv, load_model, load_json
from .features import build_features


def inverse_target_transform(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(y)
    return y


def forecast(
    data_path: str | Path,
    artifacts_dir: str | Path,
    output_path: str | Path,
    feature_cfg: FeatureConfig,
):
    artifacts_dir = Path(artifacts_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    df = load_csv(data_path)
    print(f"[FORECAST] Loaded data: {df.shape}")

    # 2) Load artifacts
    model = load_model(artifacts_dir / "model.joblib")
    series_profile = pd.read_parquet(artifacts_dir / "series_profile.parquet")

    features_meta = load_json(artifacts_dir / "features.json")
    metadata = load_json(artifacts_dir / "metadata.json")

    raw_feature_names = features_meta["raw_feature_names"]
    safe_feature_names = features_meta["feature_names"]

    horizons = metadata["horizons"]
    transform = metadata.get("target_transform", "none")

    # 3) Compute forecast origins: last observed month per (agency, sku)
    df[feature_cfg.date_col] = pd.to_datetime(df[feature_cfg.date_col])

    last_obs = (
        df.sort_values(feature_cfg.date_col)
          .groupby(feature_cfg.id_cols, as_index=False)
          .tail(1)
    )
    print(f"[FORECAST] Found {len(last_obs)} series origins")

    # 4) Build features for ALL rows (predict mode), then keep only origins
    X_all, _, _ = build_features(
        df,
        cfg=feature_cfg,
        mode="predict",
        series_profile=series_profile,
    )

    # Keep only last origin rows (inner join)
    X_origin = X_all.merge(
        last_obs[feature_cfg.id_cols + [feature_cfg.date_col]],
        on=feature_cfg.id_cols + [feature_cfg.date_col],
        how="inner",
    )

    # 5) Prepare X for model (same features as training, same sanitized names)
    X_safe = X_origin[raw_feature_names].copy()
    X_safe.columns = safe_feature_names

    # Drop any row with NaN (should be rare if history is complete)
    mask = ~X_safe.isna().any(axis=1)
    X_safe = X_safe.loc[mask]
    X_origin = X_origin.loc[mask]

    if len(X_safe) == 0:
        raise ValueError("No valid forecast rows after feature construction (all rows contain NaN).")

    print(f"[FORECAST] Valid origin rows: {len(X_safe)}")

    # 6) Predict (model outputs in TRAIN SCALE)
    y_pred_fit = model.predict(X_safe)

    # Inverse transform to original scale if needed
    y_pred = inverse_target_transform(y_pred_fit, transform)

    # sales volume: enforce non-negative
    y_pred = np.clip(y_pred, 0.0, None)

    # 7) Build output (long format)
    records = []
    origin_dates = pd.to_datetime(X_origin[feature_cfg.date_col])

    for i, h in enumerate(horizons):
        tmp = X_origin[feature_cfg.id_cols].copy()
        tmp["forecast_date"] = (origin_dates + pd.DateOffset(months=h)).values
        tmp["horizon"] = h
        tmp["y_pred"] = y_pred[:, i]
        records.append(tmp)

    forecast_df = pd.concat(records, axis=0).reset_index(drop=True)

    # 8) Save
    forecast_df.to_csv(output_path, index=False)
    print(f"[FORECAST] Predictions saved to: {output_path}")


if __name__ == "__main__":
    forecast(
        data_path="data/ds_assortiment_dataset.csv",
        artifacts_dir="artifacts/debug_run",
        output_path="artifacts/debug_run/predictions.csv",
        feature_cfg=FeatureConfig(),
    )
