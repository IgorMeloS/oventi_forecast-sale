# src/forecast_sales/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


# =========================
# Feature configuration
# =========================

@dataclass(frozen=True)
class FeatureConfig:
    # --- Dataset columns (REAL DATASET) ---
    id_cols: List[str] = field(default_factory=lambda: ["agency", "sku"])
    date_col: str = "date"          # YYYY-MM-01 (monthly)
    target_col: str = "volume"      
    # --- Forecast setup ---
    horizons: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # --- Target transform ---
    target_transform: Literal["none", "log1p"] = "log1p"

    # --- Sales history (core signal) ---
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 12])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std"])

    # --- Calendar features ---
    use_calendar: bool = True

    # --- Intermittence / dormance ---
    use_intermittence: bool = True
    zeros_window: int = 3
    nonzero_window: int = 12
    tsls_cap: int = 24  # time since last sale (cap)

    # --- Series profile (TRAIN ONLY) ---
    use_series_profile: bool = True

    # --- Exogenous features ---
    # Price
    use_price: bool = True
    price_col: str = "price_actual"
    price_roll_windows: List[int] = field(default_factory=lambda: [3])

    # Discount
    use_discount: bool = True
    discount_col: str = "discount_in_percent"
    discount_roll_windows: List[int] = field(default_factory=lambda: [3])
    discount_clip_min: float = 0.0
    discount_clip_max: float = 100.0


# =========================
# Training configuration
# =========================

@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42

    # Minimum history to keep a row (warmup)
    min_history: int = 12

    # LightGBM base parameters (V1)
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    })
