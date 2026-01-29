from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

from .config import FeatureConfig, TrainingConfig
from .io import load_csv, save_model, save_json
from .split import temporal_train_val_test_split
from .features import build_features, fit_series_profile
from .evaluation import mae, wape


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def sanitize_feature_names(feature_names: list[str]) -> list[str]:
    """LightGBM does not support special characters in feature names."""
    return [
        name.replace("[", "_")
            .replace("]", "_")
            .replace("{", "_")
            .replace("}", "_")
            .replace(":", "_")
            .replace(",", "_")
            .replace(" ", "_")
        for name in feature_names
    ]


def drop_warmup_and_nan(X: pd.DataFrame, Y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows with NaN due to lags, rollings, or horizons."""
    mask = (~X.isna().any(axis=1)) & (~Y.isna().any(axis=1))
    return X.loc[mask], Y.loc[mask]


def apply_target_transform(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(y)
    return y


def inverse_target_transform(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(y)
    return y


def compute_metrics_multioutput(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int],
) -> dict:
    """Compute MAE and WAPE per horizon + mean across horizons."""
    out = {}
    for i, h in enumerate(horizons):
        out[f"h{h}_mae"] = float(mae(y_true[:, i], y_pred[:, i]))
        out[f"h{h}_wape"] = float(wape(y_true[:, i], y_pred[:, i]))

    out["mae_mean"] = float(np.mean([v for k, v in out.items() if k.endswith("_mae")]))
    out["wape_mean"] = float(np.mean([v for k, v in out.items() if k.endswith("_wape")]))
    return out


def metrics_to_table(metrics: dict, scope: str, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for h in horizons:
        rows.append({
            "scope": scope,
            "horizon": h,
            "mae": metrics.get(f"h{h}_mae", np.nan),
            "wape": metrics.get(f"h{h}_wape", np.nan),
        })
    rows.append({
        "scope": scope,
        "horizon": "overall",
        "mae": metrics.get("mae_mean", np.nan),
        "wape": metrics.get("wape_mean", np.nan),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train(
    data_path: str | Path,
    artifacts_dir: str | Path,
    feature_cfg: FeatureConfig,
    training_cfg: TrainingConfig,
):
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    df = load_csv(data_path)
    print(f"[TRAIN] Loaded data: {df.shape}")

    # 2) Temporal split (train/test only to keep V1 simple & avoid empty val)
    df_train, _, df_test = temporal_train_val_test_split(
        df,
        date_col=feature_cfg.date_col,
        horizon=max(feature_cfg.horizons),
        val_size=0,
        test_size=4,
    )
    print(f"[TRAIN] Train raw: {df_train.shape}")
    print(f"[TRAIN] Test raw:  {df_test.shape}")

    # 3) Series profile (TRAIN ONLY)
    series_profile = fit_series_profile(df_train, feature_cfg)
    series_profile.to_parquet(artifacts_dir / "series_profile.parquet", index=False)

    # 4) Build features
    X_train, Y_train, meta = build_features(
        df_train, cfg=feature_cfg, mode="train", series_profile=series_profile
    )
    X_test, Y_test, _ = build_features(
        df_test, cfg=feature_cfg, mode="train", series_profile=series_profile
    )

    # 5) Drop warmup rows (lags/rollings/horizons)
    X_train, Y_train = drop_warmup_and_nan(X_train, Y_train)
    X_test, Y_test = drop_warmup_and_nan(X_test, Y_test)

    print(f"[TRAIN] X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"[TRAIN] X_test:  {X_test.shape}, Y_test:  {Y_test.shape}")

    # 6) Prepare X for LightGBM (sanitize feature names)
    raw_features = meta["feature_names"]
    safe_features = sanitize_feature_names(raw_features)

    X_train_safe = X_train[raw_features].copy()
    X_train_safe.columns = safe_features

    X_test_safe = X_test[raw_features].copy()
    X_test_safe.columns = safe_features

    # 7) Apply target transform consistently (TRAIN SCALE)
    transform = feature_cfg.target_transform
    Y_train_fit = apply_target_transform(Y_train.values, transform)
    Y_test_fit = apply_target_transform(Y_test.values, transform) if len(Y_test) > 0 else None

    # 8) Train model
    model = MultiOutputRegressor(LGBMRegressor(**training_cfg.lgbm_params))
    model.fit(X_train_safe, Y_train_fit)

    # 9) Evaluate (TEST if available, else TRAIN) — report metrics on ORIGINAL SCALE
    if len(X_test_safe) > 0:
        scope = "test"
        y_true_fit = Y_test_fit
        X_eval = X_test_safe
    else:
        print("[EVAL] No valid test samples after warmup. Using TRAIN metrics.")
        scope = "train"
        y_true_fit = Y_train_fit
        X_eval = X_train_safe

    y_pred_fit = model.predict(X_eval)

    y_true = inverse_target_transform(y_true_fit, transform)
    y_pred = inverse_target_transform(y_pred_fit, transform)

    # Clip predictions to non-negative for sales volume
    y_pred = np.clip(y_pred, 0.0, None)

    metrics = compute_metrics_multioutput(y_true, y_pred, feature_cfg.horizons)

    print(f"[{scope.upper()} METRICS]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 10) Save artifacts
    save_model(model, artifacts_dir / "model.joblib")

    save_json(
        {
            "raw_feature_names": raw_features,
            "feature_names": safe_features,
            "categorical_features": feature_cfg.id_cols,
        },
        artifacts_dir / "features.json",
    )

    metadata = {
        "scope": scope,
        "horizons": feature_cfg.horizons,
        "target_transform": transform,
        "date_col": feature_cfg.date_col,
        "target_col": feature_cfg.target_col,
        "id_cols": feature_cfg.id_cols,
        "n_series": int(series_profile.shape[0]),
        "n_train_samples": int(len(X_train_safe)),
        "n_test_samples": int(len(X_test_safe)),
        "lgbm_params": training_cfg.lgbm_params,
    }
    save_json(metadata, artifacts_dir / "metadata.json")
    save_json(metrics, artifacts_dir / "metrics.json")

    metrics_df = metrics_to_table(metrics, scope, feature_cfg.horizons)
    metrics_df.to_csv(artifacts_dir / "metrics.csv", index=False)

    print(f"[TRAIN] Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    train(
        data_path="data/ds_assortiment_dataset.csv",
        artifacts_dir="artifacts/debug_run",
        feature_cfg=FeatureConfig(),
        training_cfg=TrainingConfig(),
    )
