"""
nowcast.py — NowcastMY Core Model
===================================
This script is modified by the autoresearch-lite agent loop.
It loads the prepared data, builds features, fits a model,
evaluates using expanding-window cross-validation, and prints
metrics to stdout for the agent to parse.

The agent modifies this file via DESCRIPTION / OLD / NEW diffs.
After each modification, the agent loop runs this script and
checks if RMSE improved.

IMPORTANT: The final lines of stdout MUST contain:
    RMSE_VS_AR1=<value>
    COVERAGE_90PCT=<value>
These are parsed by the agent loop to decide keep/revert.

Usage:
    python nowcast.py                    # Run with current configuration
    python nowcast.py --data-dir ./data  # Custom data directory
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================================
# CONFIGURATION (agent modifies this section)
# ============================================================================

# Model configuration
MODEL_TYPE = "ridge"          # Options: ridge, lasso, elasticnet, rf, xgboost, lgbm
MODEL_ALPHA = 1.0             # Regularization strength for Ridge/LASSO
TARGET_COL = "gdp_real"       # Target: gdp_real (YoY growth)

# Feature configuration
FEATURE_COLS = [
    "ipi",                    # Industrial Production Index YoY growth
]

# Lag configuration: how many quarters of lag for each feature
N_LAGS = 1                   # Number of lagged quarters to include (0 = current only)

# Feature transforms
USE_SURPRISE = False          # Use surprise features (actual minus previous)
USE_GROWTH_RATE = False       # Apply YoY growth rate transform

# Evaluation
MIN_TRAIN_QUARTERS = 40       # Minimum training window
EXCLUDE_COVID = False         # Exclude 2020Q1-2021Q2 from scoring

# Prediction intervals
BOOTSTRAP_N = 200             # Number of bootstrap samples for prediction intervals
CONFIDENCE_LEVEL = 0.90       # 90% prediction interval

# ============================================================================
# AR(1) BASELINE (do not modify — this is the reference)
# ============================================================================

def ar1_baseline(y: np.ndarray, min_train: int):
    """AR(1) expanding-window evaluation. Returns RMSE."""
    errors = []
    for t in range(min_train, len(y)):
        y_train = y[:t]
        x = y_train[:-1]
        y_dep = y_train[1:]
        if len(x) < 2:
            continue
        x_mean, y_mean = np.mean(x), np.mean(y_dep)
        cov = np.mean((x - x_mean) * (y_dep - y_mean))
        var = np.mean((x - x_mean) ** 2)
        b = cov / var if var > 1e-12 else 0.0
        a = y_mean - b * x_mean
        pred = a + b * y_train[-1]
        errors.append((y[t] - pred) ** 2)
    return np.sqrt(np.mean(errors)) if errors else np.nan


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features(df: pd.DataFrame, feature_cols: list, n_lags: int,
                   use_surprise: bool) -> pd.DataFrame:
    """
    Build feature matrix from the quarterly panel.
    Returns a DataFrame with date, target, and all feature columns.
    """
    result = df[["date", TARGET_COL]].copy()

    for col in feature_cols:
        # Base feature
        if col in df.columns:
            result[col] = df[col].values
        elif f"{col}_surprise" in df.columns and use_surprise:
            result[col] = df[f"{col}_surprise"].values
        else:
            continue

        # Surprise variant
        if use_surprise and f"{col}_surprise" in df.columns:
            result[f"{col}_surp"] = df[f"{col}_surprise"].values

        # Lagged features
        for lag in range(1, n_lags + 1):
            if col in result.columns:
                result[f"{col}_lag{lag}"] = result[col].shift(lag)
            if use_surprise and f"{col}_surp" in result.columns:
                result[f"{col}_surp_lag{lag}"] = result[f"{col}_surp"].shift(lag)

    return result


# ============================================================================
# MODEL FITTING
# ============================================================================

def get_model():
    """Return the configured model instance."""
    if MODEL_TYPE == "ridge":
        return Ridge(alpha=MODEL_ALPHA)
    elif MODEL_TYPE == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=MODEL_ALPHA, max_iter=10000)
    elif MODEL_TYPE == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=MODEL_ALPHA, l1_ratio=0.5, max_iter=10000)
    elif MODEL_TYPE == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    elif MODEL_TYPE == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                           random_state=42, verbosity=0)
    elif MODEL_TYPE == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                            random_state=42, verbosity=-1)
    else:
        return Ridge(alpha=MODEL_ALPHA)


# ============================================================================
# EXPANDING WINDOW EVALUATION WITH BOOTSTRAP INTERVALS
# ============================================================================

def expanding_window_eval(features_df: pd.DataFrame, min_train: int,
                          exclude_covid: bool, bootstrap_n: int,
                          confidence_level: float):
    """
    Expanding window evaluation with bootstrap prediction intervals.

    Returns dict with all metrics needed for the agent loop.
    """
    # Identify feature columns (everything except date and target)
    feature_names = [c for c in features_df.columns if c not in ["date", TARGET_COL]]

    if not feature_names:
        return None

    # Drop rows with any NaN in features or target
    clean = features_df.dropna(subset=[TARGET_COL] + feature_names).reset_index(drop=True)

    if len(clean) <= min_train:
        return None

    y_all = clean[TARGET_COL].values
    X_all = clean[feature_names].values
    dates = clean["date"].values

    # COVID quarters
    covid_start = pd.Timestamp("2020-01-01")
    covid_end = pd.Timestamp("2021-07-01")
    covid_mask = np.array([
        covid_start <= pd.Timestamp(d) < covid_end for d in dates
    ])

    predictions = []
    actuals = []
    intervals_lower = []
    intervals_upper = []
    pred_dates = []

    for t in range(min_train, len(y_all)):
        X_train = X_all[:t]
        y_train = y_all[:t]
        X_test = X_all[t:t+1]
        y_actual = y_all[t]

        # Fit model
        model = get_model()
        try:
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)[0]
        except Exception:
            y_hat = np.mean(y_train)

        # Bootstrap prediction interval
        if bootstrap_n > 0:
            boot_preds = []
            rng = np.random.RandomState(t)
            for _ in range(bootstrap_n):
                idx = rng.choice(len(y_train), size=len(y_train), replace=True)
                X_boot = X_train[idx]
                y_boot = y_train[idx]
                try:
                    m = get_model()
                    m.fit(X_boot, y_boot)
                    boot_preds.append(m.predict(X_test)[0])
                except Exception:
                    boot_preds.append(y_hat)

            boot_preds = np.array(boot_preds)
            # Add residual noise from training
            residuals = y_train - model.predict(X_train)
            residual_std = np.std(residuals) if len(residuals) > 1 else 0
            boot_preds += rng.normal(0, residual_std, len(boot_preds))

            alpha = 1 - confidence_level
            lower = np.percentile(boot_preds, 100 * alpha / 2)
            upper = np.percentile(boot_preds, 100 * (1 - alpha / 2))
        else:
            lower, upper = np.nan, np.nan

        predictions.append(y_hat)
        actuals.append(y_actual)
        intervals_lower.append(lower)
        intervals_upper.append(upper)
        pred_dates.append(dates[t])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    intervals_lower = np.array(intervals_lower)
    intervals_upper = np.array(intervals_upper)

    # Compute errors
    errors = actuals - predictions

    # Scoring mask (optionally exclude COVID)
    if exclude_covid:
        score_mask = np.array([
            not (covid_start <= pd.Timestamp(d) < covid_end) for d in pred_dates
        ])
    else:
        score_mask = np.ones(len(errors), dtype=bool)

    scored_errors = errors[score_mask]

    rmse = np.sqrt(np.mean(scored_errors ** 2)) if len(scored_errors) > 0 else np.nan
    mae = np.mean(np.abs(scored_errors)) if len(scored_errors) > 0 else np.nan

    # Directional accuracy
    dir_correct = 0
    dir_total = 0
    for i in range(len(predictions)):
        t = min_train + i
        if t > 0 and score_mask[i]:
            actual_change = actuals[i] - y_all[t - 1]
            pred_change = predictions[i] - y_all[t - 1]
            if actual_change * pred_change > 0:
                dir_correct += 1
            dir_total += 1

    dir_accuracy = dir_correct / dir_total if dir_total > 0 else np.nan

    # Interval coverage
    if not np.all(np.isnan(intervals_lower)):
        in_interval = ((actuals >= intervals_lower) & (actuals <= intervals_upper))
        scored_in_interval = in_interval[score_mask]
        coverage = np.mean(scored_in_interval) if len(scored_in_interval) > 0 else np.nan
    else:
        coverage = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_accuracy,
        "coverage": coverage,
        "n_predictions": len(predictions),
        "n_scored": int(score_mask.sum()),
        "feature_names": feature_names,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NowcastMY — GDP nowcasting model")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    panel_path = data_dir / "panel_quarterly.parquet"

    if not panel_path.exists():
        print(f"ERROR: {panel_path} not found. Run prepare_data.py first.")
        sys.exit(1)

    # Load data
    df = pd.read_parquet(panel_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Data: {len(df)} quarters, {len(df.columns)} columns")
    print(f"Target: {TARGET_COL}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Model: {MODEL_TYPE} (alpha={MODEL_ALPHA})")
    print(f"Lags: {N_LAGS}, Surprise: {USE_SURPRISE}")
    print()

    # Build features
    features_df = build_features(df, FEATURE_COLS, N_LAGS, USE_SURPRISE)

    # Evaluate
    result = expanding_window_eval(
        features_df,
        min_train=MIN_TRAIN_QUARTERS,
        exclude_covid=EXCLUDE_COVID,
        bootstrap_n=BOOTSTRAP_N,
        confidence_level=CONFIDENCE_LEVEL,
    )

    if result is None:
        print("ERROR: Evaluation failed (insufficient data or features)")
        print("RMSE_VS_AR1=999.0")
        print("COVERAGE_90PCT=nan")
        sys.exit(1)

    # Compute AR(1) baseline RMSE for ratio
    target_values = df[TARGET_COL].dropna().values
    ar1_rmse = ar1_baseline(target_values, MIN_TRAIN_QUARTERS)

    rmse_ratio = result["rmse"] / ar1_rmse if ar1_rmse > 0 else np.nan

    # Print results
    print("=" * 50)
    print(f"  RMSE:                {result['rmse']:.4f}")
    print(f"  MAE:                 {result['mae']:.4f}")
    print(f"  AR(1) RMSE:          {ar1_rmse:.4f}")
    print(f"  RMSE ratio vs AR(1): {rmse_ratio:.4f}")
    print(f"  Directional acc:     {result['directional_accuracy']:.1%}")
    print(f"  90% interval cov:    {result['coverage']:.1%}" if not np.isnan(result['coverage']) else f"  90% interval cov:    N/A")
    print(f"  N predictions:       {result['n_predictions']}")
    print(f"  Features used:       {result['feature_names']}")
    print("=" * 50)

    # These lines are parsed by the agent loop — DO NOT MODIFY FORMAT
    print(f"RMSE_VS_AR1={rmse_ratio:.6f}")
    print(f"COVERAGE_90PCT={result['coverage']:.4f}" if not np.isnan(result['coverage']) else "COVERAGE_90PCT=nan")


if __name__ == "__main__":
    main()
