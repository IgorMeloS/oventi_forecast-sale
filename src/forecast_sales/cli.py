from __future__ import annotations

import argparse
from pathlib import Path

from .config import FeatureConfig, TrainingConfig
from .training import train as train_pipeline
from .forecasting import forecast as forecast_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast_sales", description="Retail forecasting CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------------
    # train
    # -----------------------
    train_p = subparsers.add_parser("train", help="Train multi-horizon model and save artifacts")
    train_p.add_argument("--data-path", type=str, required=True, help="Path to input CSV dataset")
    train_p.add_argument("--artifacts-dir", type=str, required=True, help="Directory to save training artifacts")

    # Optional overrides (keep minimal for V1)
    train_p.add_argument(
        "--target-transform",
        type=str,
        default=None,
        choices=["none", "log1p"],
        help="Override target transform (default from config.py)",
    )

    # -----------------------
    # predict
    # -----------------------
    pred_p = subparsers.add_parser("predict", help="Generate forecasts using saved artifacts")
    pred_p.add_argument("--data-path", type=str, required=True, help="Path to input CSV dataset (full history)")
    pred_p.add_argument("--artifacts-dir", type=str, required=True, help="Directory containing training artifacts")
    pred_p.add_argument("--output-path", type=str, required=True, help="Output CSV path for predictions")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load default configs
    feature_cfg = FeatureConfig()
    training_cfg = TrainingConfig()

    # Apply CLI overrides
    if args.command == "train" and args.target_transform is not None:
        # dataclass is frozen => recreate with override
        feature_cfg = FeatureConfig(
            id_cols=feature_cfg.id_cols,
            date_col=feature_cfg.date_col,
            target_col=feature_cfg.target_col,
            horizons=feature_cfg.horizons,
            target_transform=args.target_transform,
            lags=feature_cfg.lags,
            rolling_windows=feature_cfg.rolling_windows,
            rolling_stats=feature_cfg.rolling_stats,
            use_calendar=feature_cfg.use_calendar,
            use_intermittence=feature_cfg.use_intermittence,
            zeros_window=feature_cfg.zeros_window,
            nonzero_window=feature_cfg.nonzero_window,
            tsls_cap=feature_cfg.tsls_cap,
            use_series_profile=feature_cfg.use_series_profile,
            use_price=feature_cfg.use_price,
            price_col=feature_cfg.price_col,
            price_roll_windows=feature_cfg.price_roll_windows,
            use_discount=feature_cfg.use_discount,
            discount_col=feature_cfg.discount_col,
            discount_roll_windows=feature_cfg.discount_roll_windows,
            discount_clip_min=feature_cfg.discount_clip_min,
            discount_clip_max=feature_cfg.discount_clip_max,
        )

    # Dispatch
    if args.command == "train":
        train_pipeline(
            data_path=Path(args.data_path),
            artifacts_dir=Path(args.artifacts_dir),
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
        )

    elif args.command == "predict":
        forecast_pipeline(
            data_path=Path(args.data_path),
            artifacts_dir=Path(args.artifacts_dir),
            output_path=Path(args.output_path),
            feature_cfg=feature_cfg,
        )


if __name__ == "__main__":
    main()
