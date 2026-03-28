# NowcastMY

This is an experiment to have an LLM agent autonomously improve a macroeconomic nowcasting model for Malaysia.

Instead of optimising a neural network's val_bpb, you are optimising a GDP forecasting model's RMSE ratio versus an AR(1) baseline, using open data from DOSM (Department of Statistics Malaysia).

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar28`). The branch `nowcast/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b nowcast/<tag>` from current master.
3. **Read the in-scope files**:
   - `nowcast.py` — the file you modify. Features, model config, lags, transforms.
   - `prepare_data.py` — fixed data pipeline. Downloads DOSM parquets, builds panels. **Do not modify.**
   - `baselines.py` — AR(1), AR(2), random walk baselines. **Do not modify.**
4. **Verify data exists**: Check that `./data/panel_quarterly.parquet` exists. If not, run `python prepare_data.py`.
5. **Run the baseline**: `python baselines.py` and note the AR(1) RMSE. This is the number to beat.
6. **Initialize results.tsv**: Create `results.tsv` with the header row. The baseline is already recorded.
7. **Confirm and go**: Confirm setup looks good, then start the experiment loop.

## What you are nowcasting

**Target**: Malaysia's quarterly real GDP year-on-year growth rate (column `gdp_real` in the panel).

**Problem**: GDP is published ~45 days after the quarter ends. But monthly data (IPI, CPI, trade, retail) arrives 18–40 days after each month. By the time GDP is released, 3 months of faster data already exist. Your model uses this faster data to predict GDP before the official number comes out.

**Baseline to beat**: AR(1) — predicts next quarter's GDP growth as a linear function of the previous quarter's. Anything with RMSE ratio < 1.0 beats this baseline. The PRD target is < 0.80.

## Malaysian economy: what you need to know

Use this domain knowledge when choosing features and interpreting results:

**GDP composition by sector** (production side):
- Services: ~54% of GDP (wholesale/retail trade, finance, real estate, transport)
- Manufacturing: ~25% (E&E, petroleum, chemicals, food processing)
- Mining & quarrying: ~9% (crude oil, natural gas, tin)
- Agriculture: ~9% (palm oil, rubber, fisheries, forestry)
- Construction: ~4%

**GDP composition by expenditure** (demand side):
- Private consumption (C): ~60% — best proxy is IOWRT (retail trade)
- Gross fixed capital formation (I): ~22%
- Government consumption (G): ~12%
- Net exports (X-M): ~6% — Malaysia is very trade-open (trade/GDP > 120%)

**Key relationships between predictors and GDP components**:
- IPI (industrial production) → covers mining + manufacturing + electricity = ~35% of GDP
- IPI by section: section B = mining, section C = manufacturing, section D = electricity
- IOWRT (wholesale & retail trade) → closest proxy for private consumption (~60% of GDP)
- External trade (exports + imports) → net exports component; also signals global demand
- CPI/PPI → demand/supply pressure indicators; not direct GDP components but correlated
- MEI composite → DOSM's own leading/coincident/lagging indices
- FX rates (USD/MYR) → forward-looking; ringgit strength reflects external demand expectations
- Interest rates (OPR) → BNM's policy rate; stable at 3.0% since mid-2023 but spread matters

**Seasonality**:
- Ramadan (moves ~11 days earlier each year) — depresses retail, boosts food prices
- Chinese New Year (Jan/Feb) — front-loads consumption, disrupts production
- Palm oil harvest cycle (peaks around Q4)
- Year-end government spending push (Q4)
- These effects make raw QoQ comparisons noisy; YoY growth naturally adjusts for fixed-calendar seasonality, but moving holidays still cause volatility

**COVID warning**: Quarters 2020-Q1 through 2021-Q2 contain extreme outliers (-17% to +16% GDP growth). These are real data points but can dominate RMSE. The evaluation script has an option to exclude COVID quarters from scoring (`EXCLUDE_COVID = True`). Try both settings.

## What you CAN do

Modify `nowcast.py` — this is the only file you edit. Specifically, you can change:

**Features** (the `FEATURE_COLS` list):
- `ipi` — Industrial Production Index YoY growth (headline)
- `cpi` — CPI inflation (headline, division 00)
- `cpi_core` — Core CPI inflation
- `ppi` — Producer Price Index YoY growth
- `iowrt` — Wholesale & retail trade YoY growth
- `trade_exports`, `trade_imports`, `trade_balance` — external trade aggregates
- `mei_leading`, `mei_coincident`, `mei_lagging` — composite indices (names may vary)
- `fx_usdmyr` — USD/MYR exchange rate (monthly average)
- `ir_opr`, `ir_fd_3m`, `ir_blr` — interest rate series (names may vary)
- `ipi_1d_b`, `ipi_1d_c`, `ipi_1d_d` — IPI by section: B=mining, C=manufacturing, D=electricity
- Any column ending in `_surprise` — surprise features (actual minus previous value)
- Any column ending in `_lag1`, `_lag2` etc. — lagged features (built by `build_features()`)

**Model type** (`MODEL_TYPE`):
- `ridge` — Ridge regression (start here)
- `lasso` — LASSO (L1 regularisation, automatic feature selection)
- `elasticnet` — Elastic Net (L1 + L2)
- `rf` — Random Forest
- `xgboost` — XGBoost gradient boosting
- `lgbm` — LightGBM gradient boosting

**Model hyperparameters**:
- `MODEL_ALPHA` — regularisation strength for Ridge/LASSO/ElasticNet
- For tree models: edit `get_model()` to change n_estimators, max_depth, learning_rate

**Lag structure** (`N_LAGS`): 0 = current quarter only, 1 = current + previous quarter, etc.

**Feature transforms**:
- `USE_SURPRISE = True` — use surprise features (actual minus previous value) instead of/alongside raw levels
- `USE_GROWTH_RATE = True` — apply growth rate transform
- Edit `build_features()` to add cross-series ratios, interaction terms, or custom transforms

**Evaluation settings**:
- `EXCLUDE_COVID = True/False` — whether to exclude 2020-Q1 to 2021-Q2 from RMSE scoring
- `MIN_TRAIN_QUARTERS` — minimum training window (default: 40 quarters = ~10 years)
- `BOOTSTRAP_N` — number of bootstrap samples for prediction intervals
- `CONFIDENCE_LEVEL` — prediction interval width (default: 0.90 = 90%)

## What you CANNOT do

- Modify `prepare_data.py`. It is read-only.
- Modify `baselines.py`. It is read-only.
- Install new packages or add dependencies beyond what's already installed.
- Use future data to predict the past (the expanding window in `expanding_window_eval` enforces this).
- Use the target variable's own future values as features (e.g., GDP(t+1) to predict GDP(t)).
- Use more than 15 features simultaneously in the `FEATURE_COLS` list (overfitting risk on ~80 observations).
- Use deep learning models (LSTM, Transformer) — the dataset has ~80 quarterly observations, far too few.

## The goal

**Get the lowest RMSE_VS_AR1.** A value below 1.0 means you beat the AR(1) baseline. The PRD target is < 0.80. Secondary goals: directional accuracy > 80%, 90% prediction interval coverage between 85-95%.

**Simplicity criterion**: All else being equal, simpler is better. A tiny RMSE improvement that adds messy complexity is not worth it. Removing features while maintaining performance is a win. When evaluating whether to keep a change, weigh the complexity against the improvement. A 0.001 improvement from adding 5 features? Probably not worth the overfitting risk. A 0.01 improvement from switching one feature? Keep.

## Experiment strategy: phased exploration

Follow these phases in order. Do not skip ahead — each phase builds on the previous.

### Phase A: Single-predictor bridge equations (experiments 1–10)

Start simple. Test one predictor at a time against GDP to find the strongest individual signal.

1. **Experiment 1**: Run baseline `nowcast.py` as-is (IPI only, Ridge, 1 lag). Record RMSE ratio.
2. **Experiments 2-4**: Swap IPI for other strong candidates one at a time:
   - IOWRT alone (proxy for consumption, the largest GDP component)
   - Trade balance alone (external demand signal)
   - MEI coincident index alone (DOSM's own real-time GDP proxy)
3. **Experiments 5-7**: For the best single predictor, test lag orders:
   - N_LAGS = 0 (current quarter only — no lag)
   - N_LAGS = 2 (add two quarters of history)
   - N_LAGS = 3
4. **Experiments 8-10**: For the best single predictor + best lag, test model types:
   - LASSO (automatic feature selection if you have lagged features)
   - Random Forest (nonlinear relationships)

**Done condition**: You know which single predictor gives the best RMSE ratio, and at what lag.

### Phase B: Feature engineering (experiments 11–20)

Now add predictors and transforms, one at a time, measuring marginal improvement.

1. **Experiments 11-13**: Add a second predictor to the best single-predictor model:
   - Best single + CPI (demand pressure signal)
   - Best single + trade_exports (external demand)
   - Best single + fx_usdmyr (forward-looking financial signal)
2. **Experiments 14-16**: Test feature transforms:
   - `USE_SURPRISE = True` — does surprise content beat raw levels?
   - Add cross-series ratio (e.g., exports/imports ratio) by editing `build_features()`
   - Test N_LAGS = 0 with surprise features (current-quarter surprise may be all you need)
3. **Experiments 17-20**: Build up to 3-5 features:
   - Combine the best features found so far
   - Test removing the weakest feature (does RMSE improve? overfitting signal)
   - Try the financial features: fx_usdmyr, interest rate spread

**Done condition**: You have a multi-predictor model that improves on the best single-predictor.

### Phase C: Model class exploration (experiments 21–30)

With features locked, try different model types.

1. **Experiments 21-23**: Compare linear vs tree:
   - Ridge vs LASSO vs ElasticNet (with same features)
   - XGBoost with conservative settings (max_depth=2, n_estimators=50)
   - LightGBM with conservative settings
2. **Experiments 24-26**: Tune the winning model:
   - If Ridge: try alpha = 0.1, 1.0, 10.0, 100.0
   - If XGBoost: try max_depth = 2 vs 3, n_estimators = 50 vs 100
   - If LASSO: lower alpha to include more features, raise to be sparser
3. **Experiments 27-30**: Robustness checks:
   - `EXCLUDE_COVID = True` — does the ranking change?
   - `MIN_TRAIN_QUARTERS = 30` — does a shorter window help (more predictions)?
   - `MIN_TRAIN_QUARTERS = 50` — does a longer window help (more stable coefficients)?

**Done condition**: You have the best model type + hyperparameters for the feature set.

### Phase D: Prediction intervals and refinement (experiments 31–40)

1. **Experiments 31-33**: Calibrate prediction intervals:
   - Check COVERAGE_90PCT — is it between 85-95%?
   - If too narrow (< 85%): increase BOOTSTRAP_N to 500
   - If too wide (> 95%): the model may be too uncertain; try reducing features
2. **Experiments 34-36**: Stress tests:
   - Remove the weakest feature — does RMSE hold?
   - Add one more feature from the remaining pool — does it help or hurt?
   - Try a completely different feature set (wild card experiment)
3. **Experiments 37-40**: Final optimisation:
   - Fine-tune alpha/hyperparameters around the best values
   - Test the simplest model that gets within 0.01 of best RMSE ratio (simplicity wins)
   - Record the final configuration

**Done condition**: You have a well-calibrated model with good RMSE ratio and interval coverage.

### Phase E: Wild cards (experiments 41+)

If you've exhausted Phases A-D and want to keep improving:
- Try interaction terms between features (e.g., IPI × trade_balance)
- Try a stacking ensemble of Ridge + XGBoost
- Try excluding specific quarters (e.g., election quarters, flood quarters)
- Try using GDP by sector data as additional features (gdp_supply columns)
- Try using absolute values instead of growth rates
- Every 10th experiment, try something completely different from what's worked so far (avoid local optima)

## Output format

When `nowcast.py` finishes, it prints a summary and two parseable lines:

```
==================================================
  RMSE:                1.7715
  MAE:                 1.1445
  AR(1) RMSE:          4.8146
  RMSE ratio vs AR(1): 0.3679
  Directional acc:     76.9%
  90% interval cov:    87.2%
  N predictions:       39
  Features used:       ['ipi', 'ipi_lag1']
==================================================
RMSE_VS_AR1=0.367933
COVERAGE_90PCT=0.8718
```

Extract the key metric:
```
grep "^RMSE_VS_AR1=" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and these columns:

```
experiment_id	timestamp	description	nowcast_mode	features_used	model_type	n_lags	ragged_edge_method	rmse_m1	rmse_m2	rmse_m3	rmse_avg	mae	rmse_vs_ar1	coverage_90pct	status
```

- `experiment_id`: sequential integer (0 = baseline, 1 = first experiment, etc.)
- `timestamp`: ISO format
- `description`: short text of what this experiment tried
- `nowcast_mode`: `direct` (always, for now)
- `features_used`: comma-separated list of feature columns
- `model_type`: ridge, lasso, xgboost, etc.
- `n_lags`: number of lagged quarters
- `ragged_edge_method`: `forward_fill` (default)
- `rmse_m1`, `rmse_m2`, `rmse_m3`: RMSE at each horizon (use same value for now since we evaluate quarterly)
- `rmse_avg`: average RMSE across horizons
- `mae`: mean absolute error
- `rmse_vs_ar1`: the key metric — RMSE divided by AR(1) RMSE
- `coverage_90pct`: proportion of actuals within 90% prediction interval
- `status`: `keep`, `discard`, or `crash`

Example:
```
0	2026-03-28T10:00:00	baseline: IPI only Ridge	direct	ipi	ridge	1	forward_fill	1.77	1.77	1.77	1.77	1.14	0.368	0.872	keep
1	2026-03-28T10:01:00	swap IPI for IOWRT	direct	iowrt	ridge	1	forward_fill	1.92	1.92	1.92	1.92	1.30	0.399	0.846	discard
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `nowcast/mar28`).

LOOP FOREVER:

1. Look at the current `nowcast.py` configuration and `results.tsv` history
2. Choose the next experiment following the phased strategy above
3. Edit `nowcast.py` with the experimental change
4. `git commit -am "experiment N: description"`
5. Run: `python nowcast.py --data-dir ./data > run.log 2>&1`
6. Read results: `grep "^RMSE_VS_AR1=\|^COVERAGE_90PCT=" run.log`
7. If grep is empty, the run crashed. Run `tail -n 30 run.log` for the traceback and fix.
8. Record results in `results.tsv`
9. If RMSE_VS_AR1 < best so far: keep (this is now the new best)
10. If RMSE_VS_AR1 >= best so far: `git checkout -- nowcast.py` (revert to best version)
11. Go to step 1

## Guardrails

These are hard rules. Violating them invalidates results:

1. **No future data**: The expanding window in `expanding_window_eval()` ensures this. Never bypass it.
2. **No target leakage**: Never use `gdp_real` values from quarter t or later as features for predicting quarter t.
3. **Max 15 features**: The `FEATURE_COLS` list must have at most 15 entries. With ~80 observations and 40 training quarters, more than 15 features risks severe overfitting.
4. **No deep learning**: Do not add LSTM, Transformer, or any neural network model. The dataset is too small.
5. **Always print metrics**: The last two lines of stdout MUST be `RMSE_VS_AR1=...` and `COVERAGE_90PCT=...`. The agent loop parses these.
6. **Commit before running**: Always `git commit` before running an experiment so you can revert cleanly.
7. **One change at a time**: Each experiment should change ONE thing. If you change features AND model type simultaneously, you won't know which helped.
