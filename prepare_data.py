"""
prepare_data.py — NowcastMY Data Pipeline
==========================================
Downloads all DOSM/data.gov.my parquet files, validates schemas,
aggregates monthly → quarterly, and produces a single merged panel
DataFrame ready for nowcasting.

Designed to run on Google Colab. No API keys needed — all data is
CC BY 4.0 from OpenDOSM.

Usage:
    python prepare_data.py                # Full pipeline
    python prepare_data.py --validate     # Schema validation only
    python prepare_data.py --cache-dir ./data  # Custom cache directory

Output:
    data/panel_quarterly.parquet  — merged panel (one row per quarter)
    data/panel_monthly.parquet    — merged monthly panel (for M1/M2/M3 horizons)
    Prints shape, date range, missing-value report to stdout.
"""

import os
import sys
import hashlib
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIGURATION: All DOSM parquet endpoints
# ============================================================================

# Base URLs
DOSM_BASE = "https://storage.dosm.gov.my"
DATAGOV_BASE = "https://storage.data.gov.my"

# --- TARGET VARIABLES (quarterly GDP) ---
TARGETS = {
    "gdp_real": {
        "url": f"{DOSM_BASE}/gdp/gdp_qtr_real.parquet",
        "frequency": "quarterly",
        "description": "Real GDP (constant 2015 prices)",
        "schema": {
            "required_cols": ["date", "series_type", "value"],
            "series_type_filter": "growth_yoy",
        },
    },
    "gdp_real_sa": {
        "url": f"{DOSM_BASE}/gdp/gdp_qtr_real_sa.parquet",
        "frequency": "quarterly",
        "description": "Real GDP, seasonally adjusted",
        "schema": {
            "required_cols": ["date", "series_type", "value"],
            "series_type_filter": "growth_qoq",
        },
    },
    "gdp_supply": {
        "url": f"{DOSM_BASE}/gdp/gdp_qtr_real_supply.parquet",
        "frequency": "quarterly",
        "description": "GDP by 5 economic sectors",
        "schema": {
            "required_cols": ["date", "series_type", "sector", "value"],
            "series_type_filter": "growth_yoy",
            "pivot_col": "sector",
        },
    },
    "gdp_demand": {
        "url": f"{DOSM_BASE}/gdp/gdp_qtr_real_demand.parquet",
        "frequency": "quarterly",
        "description": "GDP by expenditure type (C+I+G+X-M)",
        "schema": {
            "required_cols": ["date", "series_type", "demandtype", "value"],
            "series_type_filter": "growth_yoy",
            "pivot_col": "demandtype",
        },
    },
}

# --- MONTHLY PREDICTORS (fast-arriving) ---
PREDICTORS_MONTHLY = {
    "cpi": {
        "url": f"{DOSM_BASE}/cpi/cpi_2d_inflation.parquet",
        "frequency": "monthly",
        "pub_lag_days": 21,
        "description": "CPI inflation by division",
        "schema": {
            "required_cols": ["date", "division", "inflation"],
            "value_col": "inflation",
            "filter_col": "division",
            "filter_val": "00",  # Overall CPI (division code '00' = overall)
        },
    },
    "cpi_core": {
        "url": f"{DOSM_BASE}/cpi/cpi_2d_core_inflation.parquet",
        "frequency": "monthly",
        "pub_lag_days": 21,
        "description": "Core CPI inflation by division",
        "schema": {
            "required_cols": ["date"],
            "value_col": "inflation",
            "filter_col": "division",
            "filter_val": "00",
        },
    },
    "ppi": {
        "url": f"{DOSM_BASE}/ppi/ppi.parquet",
        "frequency": "monthly",
        "pub_lag_days": 24,
        "description": "Producer Price Index (headline)",
        "schema": {
            "required_cols": ["date", "series_type", "value"],
            "series_type_filter": "growth_yoy",
        },
    },
    "ipi": {
        "url": f"{DOSM_BASE}/ipi/ipi.parquet",
        "frequency": "monthly",
        "pub_lag_days": 35,
        "description": "Industrial Production Index (headline)",
        "schema": {
            "required_cols": ["date", "series_type", "value"],
            "series_type_filter": "growth_yoy",
        },
    },
    "ipi_1d": {
        "url": f"{DOSM_BASE}/ipi/ipi_1d.parquet",
        "frequency": "monthly",
        "pub_lag_days": 35,
        "description": "IPI by MSIC section",
        "schema": {
            "required_cols": ["date", "series_type", "section", "value"],
            "series_type_filter": "growth_yoy",
            "pivot_col": "section",
        },
    },
    "ipi_export": {
        "url": f"{DOSM_BASE}/ipi/ipi_export.parquet",
        "frequency": "monthly",
        "pub_lag_days": 35,
        "description": "IPI for export-oriented divisions",
        "schema": {
            "required_cols": ["date", "series_type"],
            "series_type_filter": "growth_yoy",
        },
    },
    "trade": {
        "url": f"{DOSM_BASE}/trade/trade_sitc_1d.parquet",
        "frequency": "monthly",
        "pub_lag_days": 18,
        "description": "External trade by SITC section",
        "schema": {
            "required_cols": ["date"],
        },
    },
    "iowrt": {
        "url": f"{DOSM_BASE}/iowrt/iowrt.parquet",
        "frequency": "monthly",
        "pub_lag_days": 40,
        "description": "Wholesale & retail trade (headline)",
        "schema": {
            "required_cols": ["date", "series_type", "value"],
            "series_type_filter": "growth_yoy",
        },
    },
    "mfg": {
        "url": f"{DOSM_BASE}/mfg/mfg.parquet",
        "frequency": "monthly",
        "pub_lag_days": 35,
        "description": "Manufacturing statistics (headline sales/employees/wages)",
        "schema": {
            "required_cols": ["date"],
        },
        "fallback_urls": [
            f"{DOSM_BASE}/bci/mfg.parquet",
            f"{DOSM_BASE}/manufacturing/mfg.parquet",
        ],
    },
    "lfs": {
        "url": f"{DOSM_BASE}/labour/lfs_month.parquet",
        "frequency": "monthly",
        "pub_lag_days": 40,
        "description": "Labour force survey (monthly headline)",
        "schema": {
            "required_cols": ["date"],
        },
        "fallback_urls": [
            f"{DOSM_BASE}/labour/lfs_month_status.parquet",
            f"{DOSM_BASE}/lfs/lfs.parquet",
        ],
    },
    "mei": {
        "url": f"{DOSM_BASE}/mei/mei.parquet",
        "frequency": "monthly",
        "pub_lag_days": 60,
        "description": "Leading/coincident/lagging composite indices",
        "schema": {
            "required_cols": ["date"],
        },
    },
}

# --- FINANCIAL / HIGH-FREQUENCY DATA ---
PREDICTORS_FINANCIAL = {
    "fx_daily": {
        "url": f"{DATAGOV_BASE}/finsector/exr/daily_1200.parquet",
        "frequency": "daily",
        "pub_lag_days": 1,
        "description": "FX rates at 1200 (mid-day reference)",
        "schema": {
            "required_cols": ["date"],
        },
    },
    "fx_monthly": {
        "url": f"{DATAGOV_BASE}/finsector/exr/monthly.parquet",
        "frequency": "monthly",
        "pub_lag_days": 5,
        "description": "FX rates (monthly aggregates)",
        "schema": {
            "required_cols": ["date"],
        },
    },
    "interest_rates": {
        "url": f"{DATAGOV_BASE}/finsector/interest_rates.parquet",
        "frequency": "monthly",
        "pub_lag_days": 30,
        "description": "BNM interest rates (OPR, FD, lending)",
        "schema": {
            "required_cols": ["date"],
        },
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _cache_path(url: str, cache_dir: Path) -> Path:
    """Deterministic cache filename from URL."""
    name = url.split("/")[-1]
    prefix = hashlib.md5(url.encode()).hexdigest()[:8]
    return cache_dir / f"{prefix}_{name}"


def download_parquet(name: str, url: str, cache_dir: Path, force: bool = False,
                    fallback_urls: list[str] | None = None) -> pd.DataFrame | None:
    """
    Download a parquet file, cache locally, return DataFrame.
    If primary URL fails, tries fallback_urls in order.
    Returns None if all attempts fail (with warning, not crash).
    """
    cached = _cache_path(url, cache_dir)

    if cached.exists() and not force:
        try:
            df = pd.read_parquet(cached)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass  # re-download if cache is corrupted

    urls_to_try = [url] + (fallback_urls or [])

    for attempt_url in urls_to_try:
        print(f"  Downloading {name} from {attempt_url.split('/')[-1]}... ", end="", flush=True)
        try:
            df = pd.read_parquet(attempt_url)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            # Cache locally using primary URL's cache key
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cached, index=False)
            print(f"OK ({len(df):,} rows, {len(df.columns)} cols)")
            return df
        except Exception as e:
            print(f"FAILED ({e})")
            continue

    return None


def validate_schema(name: str, df: pd.DataFrame, schema: dict) -> list[str]:
    """Check that a DataFrame matches the expected schema. Returns list of issues."""
    issues = []
    if df is None:
        return [f"{name}: DataFrame is None (download failed)"]

    required = schema.get("required_cols", [])
    for col in required:
        if col not in df.columns:
            issues.append(f"{name}: missing required column '{col}' (have: {list(df.columns)})")

    if "date" in df.columns:
        if df["date"].isna().sum() > 0:
            issues.append(f"{name}: {df['date'].isna().sum()} null dates")
        if len(df) == 0:
            issues.append(f"{name}: empty DataFrame")

    return issues


def _quarter_start(dt: pd.Timestamp) -> pd.Timestamp:
    """Map any date to the first day of its quarter."""
    q_month = ((dt.month - 1) // 3) * 3 + 1
    return pd.Timestamp(year=dt.year, month=q_month, day=1)


# ============================================================================
# EXTRACTION FUNCTIONS — one per data pattern
# ============================================================================

def extract_simple_series(df: pd.DataFrame, schema: dict, prefix: str) -> pd.DataFrame:
    """
    Extract a single time series from the standard DOSM format:
        date | series_type | value
    Filter by series_type (e.g. 'growth_yoy') and return date + renamed value column.
    """
    if df is None:
        return pd.DataFrame()

    st_filter = schema.get("series_type_filter")
    if st_filter and "series_type" in df.columns:
        df = df[df["series_type"] == st_filter].copy()

    # Handle CPI/core CPI which use 'inflation' instead of 'value' and need division filter
    filter_col = schema.get("filter_col")
    filter_val = schema.get("filter_val")
    if filter_col and filter_col in df.columns and filter_val:
        df = df[df[filter_col].astype(str) == str(filter_val)].copy()

    value_col = schema.get("value_col", "value")
    if value_col not in df.columns:
        # Try to find any numeric column that could be the value
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_date_numeric = [c for c in numeric_cols if c != "date"]
        if non_date_numeric:
            value_col = non_date_numeric[0]
        else:
            return pd.DataFrame()

    result = df[["date", value_col]].copy()
    result = result.rename(columns={value_col: prefix})
    result = result.dropna(subset=[prefix])
    result = result.drop_duplicates(subset=["date"], keep="last")
    result = result.sort_values("date").reset_index(drop=True)
    return result


def extract_pivoted_series(df: pd.DataFrame, schema: dict, prefix: str) -> pd.DataFrame:
    """
    Extract multiple series from a pivoted DOSM format:
        date | series_type | sector/section/demandtype | value
    Returns one column per unique category value, e.g. gdp_supply_services, gdp_supply_mfg.
    """
    if df is None:
        return pd.DataFrame()

    st_filter = schema.get("series_type_filter")
    if st_filter and "series_type" in df.columns:
        df = df[df["series_type"] == st_filter].copy()

    pivot_col = schema.get("pivot_col")
    if not pivot_col or pivot_col not in df.columns:
        return pd.DataFrame()

    value_col = schema.get("value_col", "value")
    if value_col not in df.columns:
        return pd.DataFrame()

    # Pivot: date as index, one column per category
    try:
        pivoted = df.pivot_table(
            index="date", columns=pivot_col, values=value_col, aggfunc="first"
        )
        # Clean column names
        pivoted.columns = [f"{prefix}_{str(c).lower().replace(' ', '_')[:20]}" for c in pivoted.columns]
        pivoted = pivoted.reset_index()
        pivoted = pivoted.sort_values("date").reset_index(drop=True)
        return pivoted
    except Exception:
        return pd.DataFrame()


def extract_trade_balance(df: pd.DataFrame, prefix: str = "trade") -> pd.DataFrame:
    """
    Extract total exports, imports, and trade balance from trade SITC data.
    The trade file has columns like: date, section, exports, imports (or similar).
    We aggregate across all SITC sections per month.
    """
    if df is None:
        return pd.DataFrame()

    # Inspect what columns are available — trade data has various formats
    cols = df.columns.tolist()
    date_col = "date"

    # Look for export/import value columns
    export_candidates = [c for c in cols if "export" in c.lower() and "value" not in c.lower()]
    import_candidates = [c for c in cols if "import" in c.lower() and "value" not in c.lower()]

    # Also try generic numeric columns
    if not export_candidates:
        export_candidates = [c for c in cols if "export" in c.lower()]
    if not import_candidates:
        import_candidates = [c for c in cols if "import" in c.lower()]

    if not export_candidates or not import_candidates:
        # Fall back: aggregate all numeric columns by date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame()
        agg = df.groupby(date_col)[numeric_cols].sum().reset_index()
        agg.columns = [date_col] + [f"{prefix}_{c}" for c in numeric_cols]
        return agg

    exp_col = export_candidates[0]
    imp_col = import_candidates[0]

    # Aggregate across all SITC sections per date
    agg = df.groupby(date_col).agg(
        exports=(exp_col, "sum"),
        imports=(imp_col, "sum"),
    ).reset_index()
    agg[f"{prefix}_balance"] = agg["exports"] - agg["imports"]
    agg[f"{prefix}_exports"] = agg["exports"]
    agg[f"{prefix}_imports"] = agg["imports"]
    agg = agg.drop(columns=["exports", "imports"])
    return agg


def extract_fx_usdmyr(df: pd.DataFrame, prefix: str = "fx") -> pd.DataFrame:
    """Extract USD/MYR mid rate from the FX daily data."""
    if df is None:
        return pd.DataFrame()

    cols = df.columns.tolist()

    # Look for USD and mid/rate columns
    usd_mask = df.apply(lambda row: row.astype(str).str.contains("usd", case=False).any(), axis=1) if "currency" in cols else pd.Series([True] * len(df))

    rate_col = None
    for candidate in ["rate_middle", "rate_mid", "mid", "rate_selling", "rate"]:
        if candidate in cols:
            rate_col = candidate
            break

    if rate_col is None:
        numeric_cols = [c for c in cols if c != "date" and df[c].dtype in [np.float64, np.int64]]
        if numeric_cols:
            rate_col = numeric_cols[0]
        else:
            return pd.DataFrame()

    # Filter for USD if there's a currency column
    if "currency" in cols:
        df = df[df["currency"].astype(str).str.lower().str.contains("usd")].copy()

    result = df[["date", rate_col]].copy()
    result = result.rename(columns={rate_col: f"{prefix}_usdmyr"})
    result = result.dropna()
    result = result.sort_values("date").reset_index(drop=True)
    return result


def extract_interest_rates(df: pd.DataFrame, prefix: str = "ir") -> pd.DataFrame:
    """Extract key interest rates — OPR and related."""
    if df is None:
        return pd.DataFrame()

    cols = df.columns.tolist()

    # Interest rate data typically has: date, variable/rate_type, value
    type_col = None
    for candidate in ["variable", "rate_type", "type", "series"]:
        if candidate in cols:
            type_col = candidate
            break

    value_col = None
    for candidate in ["value", "rate", "interest_rate"]:
        if candidate in cols:
            value_col = candidate
            break

    if type_col and value_col:
        # Pivot to get one column per rate type
        try:
            pivoted = df.pivot_table(index="date", columns=type_col, values=value_col, aggfunc="first")
            # Keep only a reasonable number of rate types (take first 10)
            if len(pivoted.columns) > 10:
                pivoted = pivoted.iloc[:, :10]
            pivoted.columns = [f"{prefix}_{str(c).lower().replace(' ', '_')[:25]}" for c in pivoted.columns]
            return pivoted.reset_index()
        except Exception:
            pass

    # Fallback: return all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols and "date" in cols:
        result = df[["date"] + numeric_cols].copy()
        result.columns = ["date"] + [f"{prefix}_{c}" for c in numeric_cols]
        return result.drop_duplicates(subset=["date"], keep="last")

    return pd.DataFrame()


def extract_mei(df: pd.DataFrame, prefix: str = "mei") -> pd.DataFrame:
    """Extract MEI leading/coincident/lagging composite indices."""
    if df is None:
        return pd.DataFrame()

    cols = df.columns.tolist()
    numeric_cols = [c for c in cols if c != "date" and c != "series_type"]

    # MEI may have: date, leading, coincident, lagging, diffusion columns
    # or: date, variable, value format
    if "variable" in cols and "value" in cols:
        try:
            pivoted = df.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
            pivoted.columns = [f"{prefix}_{str(c).lower().replace(' ', '_')[:20]}" for c in pivoted.columns]
            return pivoted.reset_index()
        except Exception:
            pass

    # Direct format: multiple columns per date
    result_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "date"]
    if result_cols and "date" in cols:
        result = df[["date"] + result_cols].copy()
        result = result.drop_duplicates(subset=["date"], keep="last")
        result.columns = ["date"] + [f"{prefix}_{c}" for c in result_cols]
        return result

    return pd.DataFrame()


def extract_lfs(df: pd.DataFrame, prefix: str = "lfs") -> pd.DataFrame:
    """Extract key labour force statistics."""
    if df is None:
        return pd.DataFrame()

    cols = df.columns.tolist()

    # LFS may have: date, variable, value (pivoted) or direct columns
    if "variable" in cols and "value" in cols:
        try:
            # Filter for key variables only
            key_vars = df["variable"].unique()[:15]  # cap at 15
            df_filtered = df[df["variable"].isin(key_vars)]
            pivoted = df_filtered.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
            pivoted.columns = [f"{prefix}_{str(c).lower().replace(' ', '_')[:25]}" for c in pivoted.columns]
            return pivoted.reset_index()
        except Exception:
            pass

    # Direct format
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols and "date" in cols:
        result = df[["date"] + numeric_cols[:10]].copy()  # cap at 10 columns
        result = result.drop_duplicates(subset=["date"], keep="last")
        result.columns = ["date"] + [f"{prefix}_{c}" for c in numeric_cols[:10]]
        return result

    return pd.DataFrame()


# ============================================================================
# AGGREGATION: Monthly → Quarterly
# ============================================================================

def monthly_to_quarterly(df: pd.DataFrame, agg_method: str = "mean") -> pd.DataFrame:
    """
    Aggregate a monthly DataFrame to quarterly frequency.
    agg_method: 'mean' (default), 'last', or 'sum'.
    All non-date columns are aggregated.
    """
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["quarter"] = df["date"].apply(_quarter_start)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    if agg_method == "mean":
        agg = df.groupby("quarter")[numeric_cols].mean()
    elif agg_method == "last":
        agg = df.groupby("quarter")[numeric_cols].last()
    elif agg_method == "sum":
        agg = df.groupby("quarter")[numeric_cols].sum()
    else:
        agg = df.groupby("quarter")[numeric_cols].mean()

    agg = agg.reset_index().rename(columns={"quarter": "date"})
    return agg


def build_monthly_availability_matrix(predictors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a matrix showing which monthly series have data for each month.
    Used for M1/M2/M3 horizon simulation.
    Columns: date, then one boolean column per predictor name.
    """
    all_dates = set()
    for df in predictors.values():
        if df is not None and not df.empty and "date" in df.columns:
            all_dates.update(df["date"].tolist())

    if not all_dates:
        return pd.DataFrame()

    all_dates = sorted(all_dates)
    matrix = pd.DataFrame({"date": all_dates})

    for name, df in predictors.items():
        if df is not None and not df.empty and "date" in df.columns:
            available_dates = set(df["date"].tolist())
            matrix[f"has_{name}"] = matrix["date"].isin(available_dates)
        else:
            matrix[f"has_{name}"] = False

    return matrix


# ============================================================================
# SURPRISE FEATURES (St. Louis Fed ENI methodology)
# ============================================================================

def compute_surprise_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    For each column, compute surprise = actual - AR(1) forecast.
    The AR(1) forecast for month t is: predicted = a + b * value(t-1),
    where a and b are estimated on all data up to t-1 (expanding window).

    For simplicity, we use a rolling approach: surprise = actual - previous value
    (equivalent to random walk surprise). The agent can refine this later.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns and col != "date":
            df[f"{col}_surprise"] = df[col] - df[col].shift(1)
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(cache_dir: Path, force_download: bool = False, validate_only: bool = False):
    """Execute the full data pipeline."""

    print("=" * 70)
    print("  NowcastMY Data Pipeline")
    print(f"  Cache directory: {cache_dir}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_issues = []

    # ------------------------------------------------------------------
    # STAGE 1: Download all parquet files
    # ------------------------------------------------------------------
    print("\n[Stage 1] Downloading parquet files...\n")

    raw_data = {}

    print("  --- Target variables (quarterly GDP) ---")
    for name, config in TARGETS.items():
        df = download_parquet(name, config["url"], cache_dir, force=force_download,
                            fallback_urls=config.get("fallback_urls"))
        raw_data[name] = df
        issues = validate_schema(name, df, config["schema"])
        all_issues.extend(issues)

    print("\n  --- Monthly predictors ---")
    for name, config in PREDICTORS_MONTHLY.items():
        df = download_parquet(name, config["url"], cache_dir, force=force_download,
                            fallback_urls=config.get("fallback_urls"))
        raw_data[name] = df
        issues = validate_schema(name, df, config["schema"])
        all_issues.extend(issues)

    print("\n  --- Financial / high-frequency data ---")
    for name, config in PREDICTORS_FINANCIAL.items():
        df = download_parquet(name, config["url"], cache_dir, force=force_download,
                            fallback_urls=config.get("fallback_urls"))
        raw_data[name] = df
        issues = validate_schema(name, df, config["schema"])
        all_issues.extend(issues)

    # ------------------------------------------------------------------
    # VALIDATION REPORT
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Schema Validation Report")
    print("=" * 70)

    successful = sum(1 for v in raw_data.values() if v is not None)
    failed = sum(1 for v in raw_data.values() if v is None)

    print(f"\n  Downloads: {successful} successful, {failed} failed")

    if all_issues:
        print(f"\n  Schema issues ({len(all_issues)}):")
        for issue in all_issues:
            print(f"    WARNING: {issue}")
    else:
        print("  Schema: All validations passed")

    # Print summary for each downloaded file
    print(f"\n  {'Name':<20} {'Rows':>8} {'Cols':>5} {'Date range':<30}")
    print("  " + "-" * 65)
    for name, df in raw_data.items():
        if df is not None:
            date_range = ""
            if "date" in df.columns:
                date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            print(f"  {name:<20} {len(df):>8,} {len(df.columns):>5} {date_range}")
        else:
            print(f"  {name:<20} {'FAILED':>8}")

    if validate_only:
        print("\n  [--validate mode] Stopping after validation.")
        return None, None

    # ------------------------------------------------------------------
    # STAGE 2: Extract clean series from each source
    # ------------------------------------------------------------------
    print("\n[Stage 2] Extracting clean series...\n")

    extracted_monthly = {}
    extracted_quarterly = {}

    # GDP targets
    for name in ["gdp_real", "gdp_real_sa"]:
        config = TARGETS[name]
        series = extract_simple_series(raw_data.get(name), config["schema"], name)
        if not series.empty:
            extracted_quarterly[name] = series
            print(f"  {name}: {len(series)} quarterly obs")

    for name in ["gdp_supply", "gdp_demand"]:
        config = TARGETS[name]
        series = extract_pivoted_series(raw_data.get(name), config["schema"], name)
        if not series.empty:
            extracted_quarterly[name] = series
            print(f"  {name}: {len(series)} quarterly obs, {len(series.columns)-1} series")

    # Monthly predictors — simple series
    for name in ["cpi", "cpi_core", "ppi", "ipi", "iowrt"]:
        if name not in PREDICTORS_MONTHLY:
            continue
        config = PREDICTORS_MONTHLY[name]
        series = extract_simple_series(raw_data.get(name), config["schema"], name)
        if not series.empty:
            extracted_monthly[name] = series
            print(f"  {name}: {len(series)} monthly obs")

    # IPI by section (pivoted)
    if raw_data.get("ipi_1d") is not None:
        series = extract_pivoted_series(raw_data["ipi_1d"], PREDICTORS_MONTHLY["ipi_1d"]["schema"], "ipi_1d")
        if not series.empty:
            extracted_monthly["ipi_1d"] = series
            print(f"  ipi_1d: {len(series)} monthly obs, {len(series.columns)-1} sections")

    # Trade
    trade_df = extract_trade_balance(raw_data.get("trade"))
    if not trade_df.empty:
        extracted_monthly["trade"] = trade_df
        print(f"  trade: {len(trade_df)} monthly obs")

    # MEI
    mei_df = extract_mei(raw_data.get("mei"))
    if not mei_df.empty:
        extracted_monthly["mei"] = mei_df
        print(f"  mei: {len(mei_df)} monthly obs, {len(mei_df.columns)-1} indices")

    # Manufacturing stats — may have various schemas depending on which URL succeeded
    if raw_data.get("mfg") is not None:
        mfg_df = raw_data["mfg"]
        series = extract_simple_series(mfg_df, PREDICTORS_MONTHLY["mfg"]["schema"], "mfg")
        if series.empty:
            series = extract_lfs(mfg_df, "mfg")  # reuse generic extractor
        if not series.empty:
            extracted_monthly["mfg"] = series
            print(f"  mfg: {len(series)} monthly obs, {len(series.columns)-1} variables")
        else:
            print(f"  mfg: downloaded but could not extract series (cols: {list(mfg_df.columns)[:8]})")
    else:
        print(f"  mfg: SKIPPED (all download attempts failed — non-critical, IPI covers manufacturing output)")

    # Labour force
    lfs_df = extract_lfs(raw_data.get("lfs"))
    if not lfs_df.empty:
        extracted_monthly["lfs"] = lfs_df
        print(f"  lfs: {len(lfs_df)} monthly obs, {len(lfs_df.columns)-1} variables")
    elif raw_data.get("lfs") is not None:
        print(f"  lfs: downloaded but could not extract series (cols: {list(raw_data['lfs'].columns)[:8]})")
    else:
        print(f"  lfs: SKIPPED (all download attempts failed — non-critical, unemployment has low variance)")

    # FX rates — aggregate daily to monthly
    fx_daily = extract_fx_usdmyr(raw_data.get("fx_daily"))
    if not fx_daily.empty:
        fx_monthly_agg = fx_daily.copy()
        fx_monthly_agg["month"] = fx_monthly_agg["date"].dt.to_period("M").dt.to_timestamp()
        fx_monthly_agg = fx_monthly_agg.groupby("month")["fx_usdmyr"].mean().reset_index()
        fx_monthly_agg = fx_monthly_agg.rename(columns={"month": "date"})
        extracted_monthly["fx"] = fx_monthly_agg
        print(f"  fx: {len(fx_monthly_agg)} monthly obs (aggregated from daily)")

    # Interest rates
    ir_df = extract_interest_rates(raw_data.get("interest_rates"))
    if not ir_df.empty:
        extracted_monthly["interest_rates"] = ir_df
        print(f"  interest_rates: {len(ir_df)} monthly obs, {len(ir_df.columns)-1} rate types")

    # ------------------------------------------------------------------
    # STAGE 3: Build monthly panel (for M1/M2/M3 horizon simulation)
    # ------------------------------------------------------------------
    print("\n[Stage 3] Building monthly panel...\n")

    monthly_panel = None
    for name, df in extracted_monthly.items():
        if df.empty:
            continue
        df = df.copy()
        if monthly_panel is None:
            monthly_panel = df
        else:
            monthly_panel = monthly_panel.merge(df, on="date", how="outer")

    if monthly_panel is not None:
        monthly_panel = monthly_panel.sort_values("date").reset_index(drop=True)
        print(f"  Monthly panel shape: {monthly_panel.shape}")
        print(f"  Date range: {monthly_panel['date'].min()} to {monthly_panel['date'].max()}")

        # Compute surprise features for all numeric columns
        numeric_cols = [c for c in monthly_panel.columns if c != "date"]
        monthly_panel = compute_surprise_features(monthly_panel, numeric_cols)
        print(f"  After surprise features: {monthly_panel.shape}")
    else:
        print("  WARNING: No monthly data extracted")
        monthly_panel = pd.DataFrame()

    # ------------------------------------------------------------------
    # STAGE 4: Aggregate monthly → quarterly, merge with GDP target
    # ------------------------------------------------------------------
    print("\n[Stage 4] Building quarterly panel...\n")

    # Aggregate monthly predictors to quarterly
    quarterly_predictors = monthly_to_quarterly(monthly_panel, agg_method="mean")
    if not quarterly_predictors.empty:
        print(f"  Quarterly predictors: {quarterly_predictors.shape}")

    # Start with the primary GDP target
    gdp_yoy = extracted_quarterly.get("gdp_real")
    if gdp_yoy is None or gdp_yoy.empty:
        print("  ERROR: No GDP target data available. Cannot build quarterly panel.")
        quarterly_panel = pd.DataFrame()
    else:
        quarterly_panel = gdp_yoy.copy()
        print(f"  GDP target (YoY): {len(quarterly_panel)} quarters")

        # Merge quarterly predictors
        if not quarterly_predictors.empty:
            quarterly_panel = quarterly_panel.merge(quarterly_predictors, on="date", how="left")

        # Merge other quarterly targets (GDP by sector, by demand)
        for name in ["gdp_supply", "gdp_demand", "gdp_real_sa"]:
            extra = extracted_quarterly.get(name)
            if extra is not None and not extra.empty:
                quarterly_panel = quarterly_panel.merge(extra, on="date", how="left")

        quarterly_panel = quarterly_panel.sort_values("date").reset_index(drop=True)
        print(f"  Final quarterly panel: {quarterly_panel.shape}")
        print(f"  Date range: {quarterly_panel['date'].min()} to {quarterly_panel['date'].max()}")

    # ------------------------------------------------------------------
    # MISSING VALUE REPORT
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Missing Value Report (quarterly panel)")
    print("=" * 70)

    if not quarterly_panel.empty:
        total_cells = quarterly_panel.shape[0] * quarterly_panel.shape[1]
        missing_cells = quarterly_panel.isna().sum().sum()
        pct_missing = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        print(f"\n  Total cells: {total_cells:,}")
        print(f"  Missing cells: {missing_cells:,} ({pct_missing:.1f}%)")
        print(f"\n  {'Column':<40} {'Missing':>8} {'% Missing':>10}")
        print("  " + "-" * 60)
        for col in quarterly_panel.columns:
            if col == "date":
                continue
            n_miss = quarterly_panel[col].isna().sum()
            pct = n_miss / len(quarterly_panel) * 100
            if n_miss > 0:
                print(f"  {col[:40]:<40} {n_miss:>8} {pct:>9.1f}%")
    else:
        print("\n  No quarterly panel to report on.")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    print("\n[Saving outputs]")

    out_dir = cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not quarterly_panel.empty:
        qpath = out_dir / "panel_quarterly.parquet"
        quarterly_panel.to_parquet(qpath, index=False)
        print(f"  Saved: {qpath} ({quarterly_panel.shape})")

    if not monthly_panel.empty:
        mpath = out_dir / "panel_monthly.parquet"
        monthly_panel.to_parquet(mpath, index=False)
        print(f"  Saved: {mpath} ({monthly_panel.shape})")

    # Also save a CSV for easy inspection
    if not quarterly_panel.empty:
        csvpath = out_dir / "panel_quarterly.csv"
        quarterly_panel.to_csv(csvpath, index=False)
        print(f"  Saved: {csvpath}")

    print("\n" + "=" * 70)
    print("  Pipeline complete!")
    print("=" * 70)

    return quarterly_panel, monthly_panel


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NowcastMY Data Pipeline — download, validate, and merge DOSM parquet data"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./data",
        help="Directory to cache downloaded parquet files (default: ./data)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Only download and validate schemas; do not build panel"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if cached files exist"
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    quarterly, monthly = run_pipeline(
        cache_dir=cache_dir,
        force_download=args.force,
        validate_only=args.validate,
    )

    if quarterly is not None and not quarterly.empty:
        print(f"\nReady to nowcast! Load with:")
        print(f"  df = pd.read_parquet('{cache_dir}/panel_quarterly.parquet')")
        sys.exit(0)
    elif args.validate:
        sys.exit(0)
    else:
        print("\nERROR: Pipeline produced empty panel. Check download errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
