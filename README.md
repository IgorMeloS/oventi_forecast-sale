# Forecast Sales — Store x Product (Monthly, 4-month horizon)

## Problem statement
We forecast monthly sales volumes for each (agency, sku) pair for the next 4 months.

The objective is not to maximize raw predictive performance, but to demonstrate a clear, production-oriented end-to-end approach covering data understanding, modeling choices, evaluation strategy, and deployment considerations.

---

## Data understanding & EDA findings

The dataset contains monthly sales histories for a fixed panel of store–product pairs.

### Dataset overview
- **Size**: ~21,000 rows, 26 columns.
- **Entities**: 58 agencies × 25 SKUs, resulting in **350 distinct time series**.
- **Time span**: January 2013 → December 2017.
- **Granularity**: strictly monthly, with dates aligned on month start.
- **Panel structure**: complete and balanced — each series contains **exactly 60 months**, with no missing periods.

This complete panel structure simplifies feature engineering and time-based backtesting, as no gap-filling is required.

### Target distribution
- Sales volumes are **right-skewed** (skewness ≈ 2.65), with heavy tails.
- Large outliers are present (p99 far above the median), likely driven by promotions, stock effects, or exceptional events.
- Zero sales exist but are **not dominant globally** (~12% of observations); however, zeros are highly concentrated in a subset of series.

### Intermittent and dormant demand
- 54 series exhibit **long zero streaks (>12 consecutive months)**.
- Among them, 49 series have **≥24 consecutive zero months**, indicating dormant or quasi-inactive products.
- This suggests the coexistence of different demand regimes (regular vs. intermittent), which may require differentiated modeling strategies.

### Data quality
- No duplicate keys detected on (agency, sku, date).
- No unparseable or inconsistent dates.
- A small number of feature anomalies were identified (e.g. negative price, discount >100%) and must be sanitized in a production pipeline.

---

## Key assumptions
- The monthly time index is complete and reliable.
- Forecast horizon is fixed to 4 months for all series.
- A **global forecasting model** (single model trained across all series) is preferred over per-series models.
- Historical price and promotion variables are assumed to be known or proxied at prediction time.

---

## Modeling implications from EDA
- The balanced panel structure supports a **global model** leveraging cross-series patterns.
- Heavy-tailed targets and zeros motivate **robust metrics** such as WAPE and MAE; MAPE is avoided.
- Intermittent and dormant series may require segmentation or alternative objectives in future iterations.
- Outliers must be handled explicitly via robust loss functions, winsorization, or promotion-aware features.
- Naive baselines provide a realistic lower bound, with WAPE ≈ 0.21 (last value) and ≈ 0.21 (seasonal naive).

---

## Feature engineering

Feature engineering is designed to be **minimal, robust, and production-oriented**, leveraging the strictly monthly and complete panel structure of the dataset.  
All features are constructed in a **causal manner**, using only information available up to the forecast origin.

### Unit of modeling
- One observation corresponds to a given `(agency, sku)` pair at a given **forecast origin month `t`**.
- The target is a **4-dimensional vector**:
  \[
  Y(t) = [y_{t+1}, y_{t+2}, y_{t+3}, y_{t+4}]
  \]
- Sales volumes may be transformed using `log1p` during training to stabilize heavy-tailed distributions; predictions are inverse-transformed at inference time.

---

### Identifier features
- `agency`
- `sku`

These identifiers allow the global model to learn cross-series patterns while remaining scalable.

---

### Sales history features
**Lagged values**
- `lag_1, lag_2, lag_3, lag_4`
- `lag_6`
- `lag_12`

**Rolling statistics** (computed on shifted values)
- Rolling mean and standard deviation over windows of 3, 6, and 12 months.

---

### Intermittence and dormancy features
- `is_zero_lag1`
- `zeros_last_3`
- `nonzero_rate_12`
- `time_since_last_sale` (capped)

---

### Calendar features
- Month encoded using sine and cosine transforms
- Quarter indicator
- Global time index (`time_idx`)

---

### Series-level profile features (train-only)
Computed using training data only and reused for inference:
- `series_mean_train`
- `series_std_train`
- `series_nonzero_rate_train`
- `series_age_train`

---

### Exogenous variables
Used conservatively to limit future uncertainty:
- Lagged price and short rolling averages
- Promotion indicators and clipped discount rolling averages

---

### Anti-leakage principles
- All features rely strictly on information available before the forecast origin.
- Series-level statistics are computed on the training split only.
- No future-dependent aggregation or target encoding is used.

---

### Scope limitations
The baseline intentionally excludes:
- Store-level aggregation
- Assortment-level features
- Probabilistic forecasts
- Learned embeddings (e.g. TFT, TSMixer)

---

## Modeling choices (high-level)
- **Baselines**: last-observation and seasonal naive.
- **Main model**: global tree-based regression (LightGBM).
- **Forecasting strategy**: direct multi-horizon (H=4).
- **Evaluation**: time-based holdout on forecast origins; rolling-origin backtesting listed as a next step.

---

## Training pipeline details

### Forecasting formulation
- Direct multi-horizon setup predicting `[y_{t+1}, y_{t+2}, y_{t+3}, y_{t+4}]`.
- Avoids recursive error accumulation.

### Model
- `MultiOutputRegressor` wrapping `LightGBMRegressor`
- Single serialized model handling all horizons.

### Warm-up handling
- Rows with insufficient historical context or missing future targets are dropped.
- Some recent periods may not yield evaluable samples.

### Artifacts
Training produces:
- `model.joblib`
- `series_profile.parquet`
- `features.json`
- `metadata.json`
- `metrics.json`

---

## Evaluation metrics

### Metrics used
- **MAE**
- **WAPE**

Metrics are reported per horizon and aggregated across horizons.

WAPE is preferred over MAPE due to intermittent demand and zero-heavy series.

---

## Installation

This project is written in **Python 3.10+**.

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
# .venv\Scripts\activate         # Windows

pip install .

---
## Deployment (conceptual – GCP)

This project is designed to be production-ready but is executed locally.

In a GCP environment, the intended setup would follow a **batch-oriented architecture**:

- **Batch training** executed on a fixed schedule (e.g. monthly), using historical sales data.
- **Batch inference** to generate 4-month-ahead forecasts for all (agency, sku) pairs.
- **Centralized artifact storage** for trained models, metrics, and prediction outputs.
- **Lightweight monitoring** on data quality and forecast performance once actuals are available.

Typical GCP components could include:
- **Cloud Storage** for datasets, model artifacts, and forecasts.
- **Cloud Run Jobs** or **Vertex AI Training** for training and inference workloads.
- **Cloud Scheduler** to orchestrate periodic runs.

Cloud deployment is intentionally kept at a conceptual level; the focus of this project is on modeling choices, evaluation strategy, and production-oriented code structure.

## Command Line Interface (CLI)

All steps are executed from the terminal.

### Training
```bash
python -m src.forecast_sales.cli train \
  --data-path data/ds_assortiment_dataset.csv \
  --artifacts-dir artifacts/run_001 \
  --target-transform log1p

### Forecasting
```bash
python -m src.forecast_sales.cli predict \
  --data-path data/ds_assortiment_dataset.csv \
  --artifacts-dir artifacts/run_001 \
  --output-path artifacts/run_001/predictions.csv

