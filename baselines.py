"""
baselines.py — NowcastMY Baseline Models
=========================================
Establishes the RMSE numbers to beat: AR(1), AR(2), and random walk baselines
evaluated using expanding-window out-of-sample forecasting at M1/M2/M3 horizons.

Reads: data/panel_quarterly.parquet (from prepare_data.py)
Writes: data/baselines_results.csv, prints results to stdout

Usage:
    python baselines.py                          # Run all baselines
    python baselines.py --data-dir ./data        # Custom data directory
    python baselines.py --min-train 40           # Minimum training quarters
    python baselines.py --target gdp_real        # Specify target column

The agent loop (nowcast.py) must produce RMSE lower than these numbers.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Publication lag (days after month-end) for each predictor group
# Used to determine what data is available at M1, M2, M3 horizons
PUB_LAGS = {
    "fast":   {"lag_days": 21,  "series": ["cpi", "cpi_core", "ppi", "fx_usdmyr"]},
    "trade":  {"lag_days": 18,  "series": ["trade_exports", "trade_imports", "trade_balance"]},
    "medium": {"lag_days": 35,  "series": ["ipi", "mfg"]},
    "slow":   {"lag_days": 40,  "series": ["iowrt", "lfs"]},
    "lagging":{"lag_days": 60,  "series": ["mei"]},
}

# Intra-quarter horizons (following GDPNow progressive updating)
# M1: ~6 weeks into quarter — only month-1 fast series available
# M2: ~10 weeks — month-1 all series + month-2 fast series
# M3: ~14 weeks — all 3 months fast, month-2 medium/slow, possibly month-3 fast
HORIZONS = {
    "M1": "Early quarter (month-1 fast data only)",
    "M2": "Mid quarter (month-1 all + month-2 fast)",
    "M3": "Late quarter (most data available, just before GDP release)",
}


# ============================================================================
# BASELINE MODELS
# ============================================================================

class RandomWalkBaseline:
    """
    GDP_growth(t) = GDP_growth(t-1)
    Pure persistence — no regression, no parameters.
    This is the simplest possible baseline, used by several Fed studies.
    """
    name = "Random Walk"

    def fit(self, y_train: np.ndarray):
        pass  # no parameters to fit

    def predict(self, y_history: np.ndarray) -> float:
        return y_history[-1] if len(y_history) > 0 else 0.0


class AR1Baseline:
    """
    GDP_growth(t) = a + b * GDP_growth(t-1) + e
    Simple autoregressive model with one lag.
    THE minimum bar for any nowcasting model.
    """
    name = "AR(1)"

    def __init__(self):
        self.a = 0.0
        self.b = 0.0

    def fit(self, y_train: np.ndarray):
        if len(y_train) < 3:
            self.a, self.b = np.mean(y_train), 0.0
            return
        y = y_train[1:]
        x = y_train[:-1]
        # OLS: y = a + b*x
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov_xy = np.mean((x - x_mean) * (y - y_mean))
        var_x = np.mean((x - x_mean) ** 2)
        if var_x > 1e-12:
            self.b = cov_xy / var_x
            self.a = y_mean - self.b * x_mean
        else:
            self.a = y_mean
            self.b = 0.0

    def predict(self, y_history: np.ndarray) -> float:
        if len(y_history) == 0:
            return self.a
        return self.a + self.b * y_history[-1]


class AR2Baseline:
    """
    GDP_growth(t) = a + b1 * GDP_growth(t-1) + b2 * GDP_growth(t-2) + e
    Two-lag autoregressive model. Slightly stronger baseline than AR(1).
    """
    name = "AR(2)"

    def __init__(self):
        self.coefs = np.zeros(3)  # [a, b1, b2]

    def fit(self, y_train: np.ndarray):
        if len(y_train) < 5:
            self.coefs = np.array([np.mean(y_train), 0.0, 0.0])
            return
        y = y_train[2:]
        X = np.column_stack([
            np.ones(len(y)),
            y_train[1:-1],
            y_train[:-2],
        ])
        # OLS via normal equations
        try:
            self.coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.coefs = np.array([np.mean(y_train), 0.0, 0.0])

    def predict(self, y_history: np.ndarray) -> float:
        if len(y_history) < 2:
            return self.coefs[0]
        return self.coefs[0] + self.coefs[1] * y_history[-1] + self.coefs[2] * y_history[-2]


class HistoricalMeanBaseline:
    """
    GDP_growth(t) = mean(GDP_growth over training window)
    Unconditional mean — the "null model" that even AR(1) should beat.
    """
    name = "Historical Mean"

    def __init__(self):
        self.mean_val = 0.0

    def fit(self, y_train: np.ndarray):
        self.mean_val = np.mean(y_train) if len(y_train) > 0 else 0.0

    def predict(self, y_history: np.ndarray) -> float:
        return self.mean_val


# ============================================================================
# EXPANDING WINDOW EVALUATION
# ============================================================================

def expanding_window_eval(y: np.ndarray, model_class, min_train: int = 40,
                          exclude_covid: bool = False, covid_quarters: list | None = None):
    """
    Expanding window out-of-sample evaluation.

    For each quarter t (starting from min_train):
      1. Train model on quarters 0..t-1
      2. Predict quarter t
      3. Record error

    Returns dict with predictions, actuals, errors, and metrics.
    """
    n = len(y)
    if n <= min_train:
        return None

    predictions = []
    actuals = []
    errors = []
    abs_errors = []

    for t in range(min_train, n):
        model = model_class()
        y_train = y[:t]

        model.fit(y_train)
        y_hat = model.predict(y_train)

        actual = y[t]
        error = actual - y_hat

        # Optionally exclude COVID quarters from metrics (but still predict them)
        is_covid = False
        if exclude_covid and covid_quarters and t in covid_quarters:
            is_covid = True

        predictions.append(y_hat)
        actuals.append(actual)
        if not is_covid:
            errors.append(error)
            abs_errors.append(abs(error))

    errors_arr = np.array(errors)
    abs_errors_arr = np.array(abs_errors)

    rmse = np.sqrt(np.mean(errors_arr ** 2)) if len(errors_arr) > 0 else np.nan
    mae = np.mean(abs_errors_arr) if len(abs_errors_arr) > 0 else np.nan

    # Directional accuracy: did we predict the right sign of change?
    dir_correct = 0
    dir_total = 0
    for i in range(len(predictions)):
        t = min_train + i
        if t > 0:
            actual_change = actuals[i] - y[t - 1]
            predicted_change = predictions[i] - y[t - 1]
            if actual_change * predicted_change > 0:
                dir_correct += 1
            dir_total += 1

    dir_accuracy = dir_correct / dir_total if dir_total > 0 else np.nan

    return {
        "predictions": np.array(predictions),
        "actuals": np.array(actuals),
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_accuracy,
        "n_predictions": len(predictions),
        "n_scored": len(errors),
    }


# ============================================================================
# MAIN
# ============================================================================

def run_baselines(data_dir: Path, target_col: str = "gdp_real", min_train: int = 40):
    """Run all baseline models and print results."""

    print("=" * 70)
    print("  NowcastMY Baseline Evaluation")
    print(f"  Target: {target_col}")
    print(f"  Min training quarters: {min_train}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    panel_path = data_dir / "panel_quarterly.parquet"
    if not panel_path.exists():
        print(f"\nERROR: {panel_path} not found. Run prepare_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(panel_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\n  Panel loaded: {df.shape[0]} quarters, {df.shape[1]} columns")
    print(f"  Date range: {df['date'].min().strftime('%Y-Q%q')} to {df['date'].max().strftime('%Y-Q%q')}"
          .replace("Q%q", f"Q{(df['date'].max().month-1)//3+1}"))

    if target_col not in df.columns:
        print(f"\n  ERROR: Target column '{target_col}' not found in panel.")
        print(f"  Available columns: {[c for c in df.columns if c != 'date'][:20]}")
        sys.exit(1)

    # Extract target series (drop NaN)
    target_df = df[["date", target_col]].dropna().reset_index(drop=True)
    y = target_df[target_col].values
    dates = target_df["date"].values

    print(f"  Target series: {len(y)} non-null observations")
    print(f"  Mean: {np.mean(y):.2f}%   Std: {np.std(y):.2f}%")
    print(f"  Min: {np.min(y):.2f}%   Max: {np.max(y):.2f}%")

    # Identify COVID quarters (2020-Q1 through 2021-Q2)
    covid_start = pd.Timestamp("2020-01-01")
    covid_end = pd.Timestamp("2021-07-01")
    covid_quarters = [
        i for i, d in enumerate(dates)
        if covid_start <= pd.Timestamp(d) < covid_end
    ]
    print(f"  COVID quarters identified: {len(covid_quarters)} (2020-Q1 to 2021-Q2)")

    # Run baselines
    models = [RandomWalkBaseline, AR1Baseline, AR2Baseline, HistoricalMeanBaseline]
    results = []

    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)

    # Full evaluation (including COVID)
    print(f"\n  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'Dir.Acc':>8} {'N':>5}")
    print("  " + "-" * 52)

    ar1_rmse = None

    for ModelClass in models:
        result = expanding_window_eval(y, ModelClass, min_train=min_train)
        if result is None:
            print(f"  {ModelClass.name:<20} {'N/A':>8} (insufficient data)")
            continue

        if ModelClass == AR1Baseline:
            ar1_rmse = result["rmse"]

        print(f"  {ModelClass.name:<20} {result['rmse']:>8.3f} {result['mae']:>8.3f} "
              f"{result['directional_accuracy']:>7.1%} {result['n_predictions']:>5}")

        results.append({
            "model": ModelClass.name,
            "variant": "full",
            "rmse": result["rmse"],
            "mae": result["mae"],
            "directional_accuracy": result["directional_accuracy"],
            "n_predictions": result["n_predictions"],
            "n_scored": result["n_scored"],
        })

    # Excluding COVID
    print(f"\n  Excluding COVID quarters (2020-Q1 to 2021-Q2):")
    print(f"  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'Dir.Acc':>8} {'N':>5}")
    print("  " + "-" * 52)

    ar1_rmse_excl = None

    for ModelClass in models:
        result = expanding_window_eval(
            y, ModelClass, min_train=min_train,
            exclude_covid=True, covid_quarters=covid_quarters
        )
        if result is None:
            continue

        if ModelClass == AR1Baseline:
            ar1_rmse_excl = result["rmse"]

        print(f"  {ModelClass.name:<20} {result['rmse']:>8.3f} {result['mae']:>8.3f} "
              f"{result['directional_accuracy']:>7.1%} {result['n_scored']:>5}")

        results.append({
            "model": ModelClass.name,
            "variant": "excl_covid",
            "rmse": result["rmse"],
            "mae": result["mae"],
            "directional_accuracy": result["directional_accuracy"],
            "n_predictions": result["n_predictions"],
            "n_scored": result["n_scored"],
        })

    # RMSE ratio summary
    if ar1_rmse is not None:
        print(f"\n  " + "=" * 52)
        print(f"  AR(1) RMSE (the number to beat): {ar1_rmse:.4f}")
        if ar1_rmse_excl is not None:
            print(f"  AR(1) RMSE (excl. COVID):        {ar1_rmse_excl:.4f}")
        print(f"\n  The agent must achieve RMSE ratio < 1.0 to beat AR(1).")
        print(f"  Target from PRD: RMSE ratio < 0.80 at M3 horizon.")

    # Save results
    results_df = pd.DataFrame(results)
    out_path = data_dir / "baselines_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Write to results.tsv (same format the agent loop uses)
    tsv_path = data_dir / "results.tsv"
    tsv_rows = []
    for _, row in results_df.iterrows():
        tsv_rows.append({
            "experiment_id": 0,
            "timestamp": datetime.now().isoformat(),
            "description": f"Baseline: {row['model']} ({row['variant']})",
            "nowcast_mode": "baseline",
            "features_used": "none (autoregressive)",
            "model_type": row["model"].lower().replace(" ", "_").replace("(", "").replace(")", ""),
            "n_lags": 1 if "AR(1)" in row["model"] else (2 if "AR(2)" in row["model"] else 0),
            "ragged_edge_method": "n/a",
            "rmse_m1": row["rmse"],  # baselines don't vary by horizon
            "rmse_m2": row["rmse"],
            "rmse_m3": row["rmse"],
            "rmse_avg": row["rmse"],
            "mae": row["mae"],
            "rmse_vs_ar1": row["rmse"] / ar1_rmse if ar1_rmse and ar1_rmse > 0 else np.nan,
            "coverage_90pct": np.nan,
            "status": "baseline",
        })

    tsv_df = pd.DataFrame(tsv_rows)
    tsv_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Saved: {tsv_path}")

    # Print the key numbers for the agent loop
    print("\n" + "=" * 70)
    print("  BASELINE RMSE (for nowcast.py stdout parsing)")
    print("=" * 70)
    if ar1_rmse is not None:
        print(f"\n  BASELINE_AR1_RMSE={ar1_rmse:.6f}")
    if ar1_rmse_excl is not None:
        print(f"  BASELINE_AR1_RMSE_EXCL_COVID={ar1_rmse_excl:.6f}")
    print()

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="NowcastMY Baselines — AR(1), AR(2), random walk evaluation"
    )
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory containing panel_quarterly.parquet")
    parser.add_argument("--target", type=str, default="gdp_real",
                       help="Target column name (default: gdp_real)")
    parser.add_argument("--min-train", type=int, default=40,
                       help="Minimum training quarters before first prediction")

    args = parser.parse_args()
    run_baselines(Path(args.data_dir), target_col=args.target, min_train=args.min_train)


if __name__ == "__main__":
    main()
