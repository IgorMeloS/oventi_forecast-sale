<!-- Project-specific Copilot instructions for AI coding agents -->
# Forecast-sales — Copilot instructions

These notes orient an AI coding agent to the repository structure, data flow, conventions and quick workflows so you can be productive without guessing project rules.

## Big picture
- Goal: predict monthly sales for each (store_id, product_id) for a fixed 4-month horizon. See `README.md` for motivation.
- Architecture: small, single-package codebase in `src/forecast_sales/` with three logical layers:
  - I/O & CLI: `io.py`, `cli.py` — CSV in/out, joblib persistence, and command-line entrypoints
  - Data & features: `preprocessing.py`, `features.py`, `split.py`, `config.py` — schema enforcement, time features, lags
  - Modeling & evaluation: `training.py`, `forecasting.py`, `evaluation.py` — training orchestration, predict flow, metrics

Changes should keep these responsibilities clear: don't move I/O into feature-engineering files, and keep persisted artifact formats stable (joblib + metadata JSON).

## Key files and what they show (use these as the primary references)
- `src/forecast_sales/config.py` — single source of truth for horizon and column names (use `CFG`).
- `src/forecast_sales/preprocessing.py` — enforces schema: required columns are `date`, `store_id`, `product_id`, `sales` by default; `date` -> pd.Timestamp and `sales` coerced to numeric with NaNs -> 0.0.
- `src/forecast_sales/features.py` — `make_time_features` (adds `year`, `month`) and `make_lag_features` (creates `lag_{n}` via groupby/shift). Ordering and sort before groupby are important.
- `src/forecast_sales/training.py` — end-to-end training: builds lags `[1,2,3]`, drops rows where lags are NA, fits a `HistGradientBoostingRegressor`, returns `(model, metrics, feature_cols)`.
- `src/forecast_sales/forecasting.py` — builds a future index using the last date in history and `CFG.horizon`. Note: current `predict()` is a minimal placeholder and does not perform iterative lag-based forecasting.
- `src/forecast_sales/io.py` — CSV load/save, model save/load via `joblib`, and `save_predictions` (creates parent dir).
- `cli.py` — exposes `train` and `predict` subcommands. Training writes `model.joblib` and a `metadata.json` containing `feature_cols` and `metrics`.

## Data and artifact conventions (discoverable rules)
- Required input columns (by default): `date`, `store_id`, `product_id`, `sales`. Use `CFG` to rename if needed.
- `date` must be parseable by `pd.to_datetime()`; the code will coerce it in `preprocess_sales`.
- Lags are created per `(store_id, product_id)` after sorting by date. Feature names: `lag_1`, `lag_2`, `lag_3` in training.
- Model artifact: `model.joblib` (joblib.dump). Metadata artifact: JSON with keys `feature_cols` (list[str]) and `metrics` (dict). CLI expects this exact shape when running `predict`.

## Developer workflows and commands
- Install (dev):
  - `python -m venv .venv` then `source .venv/bin/activate`
  - `pip install -e '.[dev]'` (project uses `pyproject.toml` dev extras with `pytest`, `ruff`)
- Run tests: `pytest` (tests are configured in pyproject: `testpaths = ["tests"]`, `pythonpath = ["src"]`).
- Run CLI (examples):
  - Train: `python -m forecast_sales.cli train --data path/to/train.csv --outdir out/artifacts`
  - Predict: `python -m forecast_sales.cli predict --data path/to/latest.csv --model out/artifacts/model.joblib --meta out/artifacts/metadata.json --out out/predictions.csv`

## Project-specific patterns & gotchas (use these when proposing code changes)
- Use `CFG` for column names/horizon. Hardcoding `"date"`/`"sales"` is fragile.
- When adding lag features, preserve the sort/group order used in `make_lag_features` (sort by group_cols + date_col) — changing this will silently produce misaligned lags.
- Training drops rows with missing lags (`df.dropna(subset=["lag_1","lag_2","lag_3"])`). If you add new lags or change lag policy, update the drop logic and tests.
- `forecasting.predict` currently does not generate iterative lag-based features for multi-step forecasting — treat it as a documented TODO if implementing true recursive forecasting. Keep backward compatibility: `predict()` currently expects `feature_cols` that may include only time features.
- Persisted `metadata.json` must include `feature_cols` used at predict time. If you change feature column names, migration or compatibility handling is required.

## Integration points / external dependencies
- Model persistence: `joblib` via `src/forecast_sales/io.py`.
- I/O: CSV read/write using pandas in `io.py` — upstream callers expect plain CSV files (no compression or exotic encodings currently handled).
- ML libs: `scikit-learn` (`HistGradientBoostingRegressor`) — keep model hyperparameters and seeds explicit if reproducibility is added.

## Where to add tests and typical small tasks
- New feature engineering -> unit tests in `tests/` that exercise `make_lag_features` and `make_time_features` with small DataFrames.
- Model/pipeline changes -> integration-like tests that run `train_model` on tiny synthetic data and assert `metadata.json` shape and metrics keys exist.

## Example quick edits an agent might do
- To implement iterative forecasting: extend `forecasting.predict` to (a) generate future rows iteratively, (b) compute lag features on the fly per-step, (c) call `model.predict` per-step. Keep the same output columns and continue saving `y_pred` as the predictions column.
- To change horizon: update `CFG.horizon` in `config.py` and ensure `make_future_index` uses it (already does).

## Final notes
- Prefer minimal, localized changes that preserve I/O contracts (`model.joblib`, `metadata.json`, CSV schemas).
- When uncertain about data shape, consult `preprocess_sales` (it documents required columns and coercions).

If anything above is unclear or you'd like more detail on a specific file or workflow, tell me which area and I'll refine this file.
