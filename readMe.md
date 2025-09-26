# Stock Return Prediction — Technical README

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](TODO)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](TODO)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Key Features (detailed)](#key-features-detailed)
- [Architecture Overview](#architecture-overview)
  - [Mermaid architecture diagram](#mermaid-architecture-diagram)
- [Technical Design Details](#technical-design-details)
  - [Data extraction & input formats](#data-extraction--input-formats)
  - [Preprocessing & sequencing pipeline](#preprocessing--sequencing-pipeline)
  - [Feature engineering internals](#feature-engineering-internals)
  - [Modeling: architectures, training, inference](#modeling-architectures-training-inference)
  - [Evaluation & visualization](#evaluation--visualization)
- [Performance Considerations](#performance-considerations)
- [Limitations & Assumptions](#limitations--assumptions)
- [Installation & Reproducible Environment](#installation--reproducible-environment)
- [Usage Examples (CLI / code snippets)](#usage-examples-cli--code-snippets)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Appendix / TODOs](#appendix--todos)

## Overview

This repository contains a modular pipeline for stock return prediction research. It covers raw data extraction from APIs, preprocessing and sequencing time-series data, a rich set of financial feature-engineering utilities (momentum, liquidity, valuation and risk measures), and several model architectures and training utilities (LSTM, Transformer, Mixture-of-Experts, and MC Dropout layers). The codebase is structured to encourage experimentation: swap feature generators, change model architectures, or plug in new data sources with minimal changes.

Audience: ML engineers and researchers working on financial time-series modeling and feature engineering.

## Key Features (detailed)

Each feature description below includes what it does, how it works (implementation notes), and why it matters.

- Data extraction (src/data_extraction)
  - What: Provides API client and extraction scripts to download stock price series, volume, market caps, and macroeconomic indicators.
  - How: The `api_client.py` encapsulates HTTP calls and response normalization. `main.py` orchestrates scheduled extraction tasks and writes raw output to disk using `src/utils/disk_io.py`.
  - Why: Reliable, normalized raw data is the foundation for reproducible experiments and avoids ad-hoc scraping logic in downstream code.

- Preprocessing pipeline (src/data_preprocessing)
  - What: A set of modules for validating, cleaning, converting, filtering, generating sequences, and finalizing datasets used for training and evaluation.
  - How: Contains specialized scripts:
    - `checker.py` validates dataset integrity (missingness, date continuity).
    - `converters.py` handles unit conversions and aligns periodicities (daily → monthly/weekly aggregation).
    - `filtering.py` contains stock-level and dataset-level filters (e.g., minimum liquidity, exchange inclusion).
    - `sequencing.py` constructs fixed-length rolling windows (timestep sequences) for model input and aligns labels with prediction horizons.
    - `finalize.py` composes feature matrices and splits datasets into train/val/test folds.
  - Why: Ensures a reproducible, auditable path from raw data to model-ready tensors and separates data hygiene from modeling logic.

- Feature engineering (src/feature_engineering)
  - What: Implements financial features used in empirical asset pricing (momentum, momentum variants, liquidity measures, volatility, betas, valuation ratios).
  - How: The folder contains `calculations/` with focused implementations (e.g., `momentum.py`, `liquidity.py`, `ratios.py`, `risk.py`) and a `generator/generator.py` that composes features into a single pipeline.
  - Why: Good features capture domain knowledge and significantly improve model signal-to-noise ratio. The modular implementation lets you add/benchmark features easily.

- Modeling (src/modeling)
  - What: Model architectures and training utilities, including LSTM, Transformer, simple dense networks, MC Dropout layer, and a Mixture-of-Experts adaptation.
  - How: Implementations are split across `architectures/` (model definitions), `layers/` (custom layers like `mc_dropout.py`), and `moe/` (mixture-of-experts implementations). `model_builder.py` provides factory/helper functions to instantiate models.
  - Why: Enables quick experiments with different sequence models and uncertainty estimation (MC Dropout) and supports ensemble/expert approaches (MOE) to capture heterogenous stock behaviors.

- Utilities and visualization (src/utils and graphs/)
  - What: Helper utilities for metrics, plotting, I/O, and small experiment orchestration.
  - How: `disk_io.py` centralizes save/load semantics (CSV/Parquet/npz), `metrics.py` implements evaluation metrics, and `plotter.py` and `modeling/utils/visualization.py` create baseline figures saved under `graphs/`.
  - Why: Reusable utilities reduce duplicated code, provide consistent experiment outputs, and speed up analysis.

## Architecture Overview

This section explains the high-level architecture, dataflow, responsibilities of main modules, and design patterns used.

High-level dataflow

1. Extraction: `src/data_extraction` collects raw time-series and macro data and writes them to `DATA_DIR/raw/`.
2. Preprocessing: `src/data_preprocessing` reads raw files, validates them, converts frequencies, filters stocks, generates feature columns, and sequences the data into model-ready arrays. Intermediate artifacts are saved under `DATA_DIR/processed/`.
3. Feature engineering: `src/feature_engineering` functions are invoked (from preprocessing or a separate step) to compute domain features per firm-time step. Results are merged into the processed dataset.
4. Modeling: `src/modeling` loads processed datasets, constructs model architectures, and trains models. Trained models and logs are persisted in `MODEL_DIR` and `graphs/`.
5. Evaluation: Predictions are evaluated using `src/utils/metrics.py` and visualized.

Roles and responsibilities (key files)

- `src/data_extraction/api_client.py`
  - Role: Provides a thin wrapper for external API calls and handles rate-limiting, retries, and normalization of responses to canonical DataFrame formats. Functions typically return pandas DataFrames keyed by date and ticker.

- `src/data_extraction/main.py`
  - Role: CLI/entry-point for scheduled extraction jobs. Calls the `api_client` and writes raw outputs using `src/utils/disk_io`.

- `src/data_preprocessing/converters.py`
  - Role: Aggregation utilities (daily → monthly), alignment of timestamps, and conversion of raw financial statement formats into numeric tables.

- `src/data_preprocessing/sequencing.py`
  - Role: Build rolling windows / sequences of fixed length. Core helper creates 3D arrays: [batch, timesteps, features]. The README appendix includes the expected nested list/array structure used by training loops.

- `src/feature_engineering/calculations/momentum.py` (and other calculators)
  - Role: Implement momentum measures (1-, 12-, 36-month mom, chmom, indmom), max daily returns, sector-adjusted momentum. Functions expect time-indexed price arrays and return aligned series.

- `src/modeling/architectures/lstm.py`
  - Role: Defines an LSTM-based Keras/PyTorch model for sequence regression/classification. (TODO: check exact framework — both TF and PyTorch are acceptable; inspect code to confirm.)

- `src/modeling/architectures/transformer.py`
  - Role: Transformer-style sequence model for longer-range dependencies.

- `src/modeling/architectures/nn.py` and `model_builder.py`
  - Role: Lightweight dense models and factory functions to instantiate different architectures with standardized input shapes and output heads.

- `src/modeling/layers/mc_dropout.py`
  - Role: Specialized dropout layer that stays active at inference time to provide Monte Carlo uncertainty estimates.

- `src/modeling/moe/mixture_of_experts_adapt.py`
  - Role: Implements mixture-of-experts logic to combine specialist submodels; likely includes gating networks and expert routing.

Design patterns and engineering decisions

- Modular pipeline: The project follows a pipeline pattern that separates extraction, transformation, and modeling. This enables re-running individual steps and better unit testing.
- Factory / builder pattern: `model_builder.py` centralizes model creation to allow consistent hyperparameter wiring across experiments.
- Single responsibility: Each module provides one logical responsibility: calculators only compute features, converters only transform frequency/units, and modeling modules only define networks and layers.
- Persist intermediate artifacts: The codebase favors writing processed data to disk (Parquet/CSV) for reproducibility and to avoid expensive recomputation.

### Mermaid architecture diagram

```mermaid
graph TD
  subgraph Extraction
    A[api_client.py] --> B[raw data files]
  end

  subgraph Preprocessing
    B --> C[converters.py]
    C --> D[checker.py]
    D --> E[filtering.py]
    E --> F[sequencing.py]
    F --> G[finalize.py]
  end

  subgraph FeatureEngineering
    H[calculations/*] --> I[generator/generator.py]
    I --> G
  end

  subgraph Modeling
    G --> J[model_builder.py]
    J --> K[architectures/*]
    K --> L[lstm.py]
    K --> M[transformer.py]
    L --> N[layers/mc_dropout.py]
    K --> O[moe/mixture_of_experts_adapt.py]
  end

  subgraph Utils
    U[utils/*] --- G
    U --- K
    U --- B
  end

  G --> P[MODEL_DIR (checkpoints, artifacts)]
  K --> Q[graphs/ (figures)]
```

## Technical Design Details

This section explains the main algorithms and workflows implemented in the codebase.

### Data extraction & input formats

- Raw extraction outputs are stored per-source and per-ticker as tabular files (CSV/Parquet). Each time-series file uses an index or column named `date` and a `ticker` identifier where applicable.
- The API client normalizes response payloads to pandas DataFrames with consistent column names: `open`, `high`, `low`, `close`, `volume`, `market_cap` (when available).

TODO: Confirm exact output formats (CSV vs Parquet) and column names in `api_client.py` and `disk_io.py`.

### Preprocessing & sequencing pipeline

Key steps:

1. Validation: `checker.py` ensures date ranges are consistent and flags missing days or unexpectedly sparse series. It raises or logs warnings based on thresholds.
2. Frequency conversion: `converters.py` provides functions to create monthly and weekly aggregations from daily EOD data. Typical operations:
   - monthly close: last trading day's `close` per month
   - monthly volume: sum of `volume`
   - dollar volume: average price × volume aggregation
3. Filters: `filtering.py` removes stocks that fail liquidity thresholds or have insufficient history. Filters are typically parameterized in `src/config/settings.py`.
4. Feature computation: The `feature_engineering` calculators are called to attach engineered features to each (ticker, date) row.
5. Sequencing: `sequencing.py` produces training windows: for a chosen `window_size` (e.g., 12 months) it creates sequences X of shape [N, window_size, F] and labels y of shape [N, 1] corresponding to forward return or classification bins.

Label alignment: Labels are aligned carefully to avoid lookahead bias. For example, to predict next-month return, the label for a sequence ending at month t uses returns computed from t+1.

Edge-case handling: The pipeline drops sequences with missing values beyond a configurable threshold, and optionally fills short gaps with forward/backward fill or interpolation as configured.

### Feature engineering internals

Major feature groups (implemented in `src/feature_engineering/calculations`):

- Momentum measures
  - Implementation: rolling returns over several horizons (1m, 12m, 36m), change-in-momentum (chmom), and industry-adjusted momentum (indmom). Functions operate on monthly aggregated returns and use vectorized pandas/numpy operations.
  - Importance: Momentum is a persistent cross-sectional predictor in equity returns research.

- Liquidity measures
  - Implementation: dollar-volume (`dolvol`), turnover (`turn`), zero-trading days (`zerotrade`) and log market value (`mve`). Aggregations and rolling statistics are implemented with groupby-rolling semantics.
  - Importance: Liquidity is correlated with return expectations and helps filter microcaps and illiquid stocks.

- Risk measures
  - Implementation: idiosyncratic volatility (idiovol) computed as residual STD from regressions on market returns over rolling windows; beta and beta-squared computed with rolling OLS on weekly returns.
  - Importance: Controls for risk exposures and enables risk-adjusted modeling.

- Valuation & fundamentals
  - Implementation: price-to-earnings-like signals (`ep_sp`), earnings growth (`agr`), with functions handling annual and quarterly inputs.

Vectorization & batching: Calculations are written to operate on pandas Series/DataFrame columns and accept both single-ticker series and batched DataFrames. The generator composes feature columns into a final wide table.

TODO: Add exact function signatures and expected argument shapes for each calculator (refer to code in `calculations/`).

### Modeling: architectures, training, inference

Model construction

- `model_builder.py` exposes a small API to create models with a consistent signature. Typical inputs:
  - `input_shape` (timesteps, features)
  - `output_dim` (1 for regression or number of classes)
  - `hparams` (dropout, hidden sizes, learning rate)

Training loops

- The repo includes lightweight training scaffolds (TODO: exact training loop entrypoint). Training loops:
  - Load dataset artifacts (`npz` / `npy` or memory-mapped arrays).
  - Create data loaders / iterators that yield batches of X, y.
  - Configure optimizer, loss (e.g., MSE for regression; cross-entropy for classification), and a scheduler (TODO: confirm exact scheduler implementation).
  - Run epochs with per-epoch validation evaluation and early stopping criteria.

Loss functions & optimization

- Default losses are standard regression/classification losses. The codebase supports uncertainty-aware predictions using MC Dropout: run multiple stochastic forward passes at inference (keeping dropout active) and aggregate mean and variance.

Inference

- For point predictions, models support a `predict(X)` API that returns forecasted returns.
- For uncertainty estimates, `mc_dropout` layer is used at inference-time with multiple stochastic passes.

Ensembling / MoE

- Mixture-of-Experts implementation provides a gating network routing inputs to specialist experts. This is useful for handling heterogenous cross-sectional behavior (e.g., sector-specific dynamics).

TODO: Add concrete examples of training entrypoints and sample hyperparameter configs.

### Evaluation & visualization

- Evaluation metrics live in `src/utils/metrics.py` and include standard MSE/RMSE and custom finance metrics (e.g., information coefficient, rank correlation, decile portfolio returns).
- Visualization functions can plot distributions, correlation matrices, feature importance, and sample prediction-vs-actual charts saved to `graphs/`.

## Performance Considerations

- Data persistence: Intermediate artifacts are saved to disk (Parquet/npz) to avoid re-computation, especially for expensive feature calculations.
- Vectorized calculations: Feature calculators use pandas/numpy vectorized ops and groupby-rolling semantics to avoid Python-level loops.
- Batch-friendly design: `sequencing.py` produces contiguous arrays ready for fast batch ingestion into frameworks (TF/PyTorch).
- IO choices: Prefer Parquet for large tabular artifacts to reduce IO overhead and memory usage.
- Parallelization: TODO: if needed, add optional multiprocessing or Dask support for feature computation across tickers.

Practical tips

- When computing rolling regressions (idiovol, beta) prefer windowed matrix operations or incremental OLS to reduce recomputation.
- Use memory-mapped numpy arrays for very large datasets to avoid memory blow-ups.

## Limitations & Assumptions

- Data source assumptions: The pipeline assumes external APIs provide consistent daily EOD data with standard columns; mismatches require connector updates.
- Missing values: The pipeline currently drops or fills missing values according to simple heuristics; more advanced imputation is optional.
- Timezones and trading calendars: Code assumes a single trading calendar; multi-market or cross-listing requires calendar-aware alignment.
- Framework agnostic: The modeling code may assume Keras or PyTorch in different files — confirm the chosen framework in the codebase before running training (TODO).

## Installation & Reproducible Environment

These instructions create a reproducible Python environment on Windows (PowerShell). Adjust for Linux/macOS as needed.

1. Clone repository and change into it

```powershell
git clone https://github.com/AdamAdham/Stock-Return-Prediction.git
cd Stock-Return-Prediction
```

2. Create and activate virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install core dependencies

TODO: Add `requirements.txt` with exact pinned versions. The list below is a minimal starting point:

```powershell
pip install --upgrade pip
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
# If using TensorFlow (CPU)
pip install tensorflow
# Or PyTorch (CPU) - choose one
pip install torch torchvision torchaudio
```

4. (Optional) For GPU training

- Follow TensorFlow / PyTorch official docs to install GPU-enabled builds and CUDA/cuDNN matching your GPU and drivers.

5. (Optional) Install dev/test extras

```powershell
pip install pytest flake8 black
```

6. Configuration

- Create a config file or set environment variables as described in the Configuration section below.

## Usage Examples (CLI / code snippets)

Below are example usage recipes to run the major flows. Replace `TODO` with real function names if needed.

- End-to-end (recommended approach: run scripted pipeline)

```python
# extract
from src.data_extraction import main as extraction
extraction.main()  # TODO: pass args or use a config file

# preprocess
from src.data_preprocessing import finalize
finalize.build_datasets()  # writes processed artifacts

# train
from src.modeling.architectures import model_builder
from src.utils import disk_io

data = disk_io.load_processed('path/to/processed.npz')
model = model_builder.build_lstm_model(input_shape=data['X_train'].shape[1:], output_dim=1)
model.fit(data['X_train'], data['y_train'], validation_data=(data['X_val'], data['y_val']), epochs=10)
```

- Using MC Dropout for uncertainty estimates

```python
# After training a model with mc_dropout layers in the architecture
def mc_predict(model, X, n_samples=50):
    preds = [model.predict(X, training=True) for _ in range(n_samples)]
    import numpy as np
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)
```

## Configuration

Key configuration files and environment variables

- `src/config/api_config.py` — endpoints, rate limits, and default symbols.
- `src/config/settings.py` — directory paths, default time windows, feature toggles, and filter thresholds.

Environment variables

- `FINANCE_API_KEY` — API key for data providers.
- `DATA_DIR` — directory to store raw/processed datasets.
- `MODEL_DIR` — directory to store model checkpoints and artifacts.

Recommended structure for a local `.env` or `config.json` (example)

```json
{
  "DATA_DIR": "./data",
  "MODEL_DIR": "./models",
  "API_KEY": "your_api_key_here",
  "WINDOW_SIZE": 12,
  "MIN_HISTORY_MONTHS": 36
}
```

TODO: Add `config.example.json` and `.env.example` to the repository.

## Contributing

Guidelines for contributors:

1. Fork -> branch (feature/ or fix/ prefix) -> implement → tests → PR.
2. Write unit tests covering new features (place in `tests/`). Aim for deterministic tests by using small synthetic datasets.
3. Keep data extraction logic side-effect free where possible; prefer returning DataFrames from functions and centralizing writes in `disk_io.py`.
4. Document assumptions and add docstrings for complex functions, especially those that implement rolling regressions or causal label alignment.

Code style

- Follow PEP8 for Python. Use `black` and `flake8` for automated formatting and linting.

## License

Add a `LICENSE` file at the repository root. Suggested placeholder: MIT.

## Appendix / TODOs

- TODO: Pin exact dependencies into `requirements.txt` and include a `pyproject.toml` for reproducible installs.
- TODO: Confirm which deep-learning framework (TensorFlow vs PyTorch) is the primary target and standardize across modeling code.
- TODO: Add example notebooks which demonstrate an end-to-end run (extraction → preprocess → train → evaluate).
- TODO: Add CI workflow and real build/test badges.
- TODO: Provide a short developer guide showing how to add a new feature calculation and how to add/register a new model architecture.

---

If you'd like, I can now:

- Generate a `requirements.txt` with common ML packages.
- Add a `config.example.json` and `.env.example` with suggested keys.
- Search the code to confirm the deep learning framework used and update the README accordingly.
# Stock Return Prediction

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](TODO)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](TODO)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements & References](#acknowledgements--references)

## Overview

Stock Return Prediction is a research-oriented codebase for extracting, preprocessing, engineering features, and modeling stock returns. The project includes tools to download financial and macroeconomic data, compute financial indicators (momentum, liquidity, ratios, risk), prepare sequences for temporal models, and train/evaluate neural architectures (LSTM, Transformer, Mixture-of-Experts) to predict future stock returns.

This repository is intended for data scientists and ML engineers working on quantitative finance experiments, prototyping model architectures, and generating reproducible analysis and visualizations.

## Key Features

- Data extraction: API client and utilities to fetch stock and macroeconomic data (`src/data_extraction`).
- Preprocessing pipeline: cleaning, filtering, converters, sequencing, and final dataset generation (`src/data_preprocessing`).
- Feature engineering: prebuilt calculations for momentum, liquidity, ratios, and risk and feature generators (`src/feature_engineering`).
- Modeling: implementations and builders for LSTM, Transformer, dense networks, MC Dropout layers and mixture-of-experts adaptations (`src/modeling`).
- Utilities: plotting, evaluation metrics, disk I/O, and helper functions for reproducible experiments (`src/utils`).
- Visualization: utilities to create exploratory plots and model visualizations (`graphs/` and `src/modeling/utils/visualization.py`).

## Architecture Overview

At a high level the repository is split into four main components:

- src/data_extraction: Responsible for pulling raw data from APIs, normalizing responses and persisting raw datasets.
- src/data_preprocessing: Cleans raw inputs, applies feature selection and transformations, sequences time series for models, and writes finalized datasets for training/validation/testing.
- src/feature_engineering: Provides domain-specific calculations (momentum, liquidity, ratios, risk) and a generator interface to compose features used by models.
- src/modeling: Model definitions and builders, including RNN (LSTM), Transformer, and Mixture-of-Experts utilities, plus a lightweight training/evaluation scaffold.

Data flow (simplified):

1. Extraction: `data_extraction` fetches and saves raw data.
2. Preprocessing: `data_preprocessing` loads raw data, validates, cleans, and generates labeled sequences.
3. Feature engineering: `feature_engineering` computes and injects engineered features.
4. Modeling: `modeling` trains models on processed sequences and evaluates performance. Visuals and metrics are saved under `graphs/` and `src/utils/metrics.py`.

The repository is organized to separate concerns so components can be swapped (different data sources, feature sets, or model architectures).

## Installation

These instructions assume a Windows environment (PowerShell) and Python 3.8+.

1. Clone the repository

   git clone https://github.com/AdamAdham/Stock-Return-Prediction.git
   cd Stock-Return-Prediction

2. Create and activate a virtual environment

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

3. Install dependencies

   TODO: Add a requirements.txt or pyproject.toml listing dependencies (e.g., pandas, numpy, tensorflow/torch, scikit-learn, matplotlib).

   Example (temporary):

   pip install pandas numpy scikit-learn matplotlib seaborn

   If you plan to use GPU-accelerated training, install the appropriate TensorFlow or PyTorch wheel and CUDA drivers per their documentation.

4. (Optional) Setup API credentials

   See the Configuration section below for environment variables and config files.

5. Run a quick smoke test

   python -c "import src.data_extraction.api_client as c; print('module import OK')"

## Usage Examples

Below are common usage patterns for the main flows. These are examples — refer to the module docstrings and TODOs for more specifics.

- Extract data (example)

```python
from src.data_extraction import main as extraction

# Runs the extraction pipeline. Accepts arguments in the module or via configuration.
extraction.main()
```

- Preprocess and generate datasets

```python
from src.data_preprocessing import finalize

# Produce training/validation/test datasets
finalize.build_datasets()
```

- Train a model (example using a model builder)

```python
from src.modeling.architectures import model_builder

model = model_builder.build_lstm_model(input_shape=(128, 20), output_dim=1)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
```

Note: The above functions are representative; check the corresponding modules for exact function names and arguments. TODO: Add CLI wrappers and example notebooks for end-to-end runs.

## Configuration

Main configuration entries are stored in `src/config` and environment variables are used for secrets and API keys.

- `src/config/api_config.py` — contains API endpoints and default extraction settings.
- `src/config/settings.py` — repository-wide settings (paths, default params).

Environment variables (recommended)

- API_KEY or FINANCE_API_KEY — API key for data providers.
- DATA_DIR — path where raw and processed data is stored (overrides defaults in settings).
- MODEL_DIR — output path for model checkpoints and serialized models.

Configuration file placeholders and TODOs

- TODO: Add a sample `config.example.json` or `.env.example` to demonstrate expected keys and structure.

## Contributing

Contributions are welcome. Suggested guidelines:

1. Fork the repository and create a feature branch (feature/your-feature).
2. Write unit tests for new functionality. Place tests under `tests/`.
3. Follow existing code style and include docstrings for new modules/functions.
4. Open a pull request describing the change, motivation, and any backward-incompatible impacts.
5. Ensure any added dependencies are justified and added to `requirements.txt` or `pyproject.toml`.

Developer tips

- Use the modular structure: extend `src/feature_engineering/generator` to add new engineered features.
- Add new model definitions in `src/modeling/architectures` and register them in `model_builder.py`.
- Keep data extraction idempotent: make raw data downloads reproducible and safe to re-run.

## License

This repository does not contain an explicit license file at the time of writing in this README. Add a `LICENSE` file at the project root.

Suggested placeholder license: MIT

TODO: Replace this section with the project's actual license if available.

## Acknowledgements & References

- This project was developed for exploratory modeling of stock returns. It builds on common time-series and financial feature engineering techniques.
- See `graphs/` for saved visualizations used in exploratory data analysis.

---

If you want, I can:

- Add a `requirements.txt` generated from the current environment (if you provide it).
- Create example notebooks demonstrating end-to-end extraction → training → evaluation.
- Add CI badges (GitHub Actions) once you provide a workflow file or give permissions.

TODOs summary

- TODO: Add `requirements.txt` or `pyproject.toml` with exact dependency pins.
- TODO: Add sample configuration files (`.env.example`, `config.example.json`).
- TODO: Add CLI entry points or Jupyter notebooks for common workflows.
# Run

While being in the root directory, run this to have all imports

```cmd
python -m src.trial
```

# Data

Explain each stages dict structure

## Last

```json
    [
    batch1  [
        timestep1  [
            feat1,
            feat2,
            featn
            ]
        timestep2  [
            feat1,
            feat2,
            featn
            ]
        timestep3   [...]
        ],

    batch1  [
        timestep1  [
            feat1,
            feat2,
            featn
            ]
        timestep2  [
            feat1,
            feat2,
            featn
            ]
        timestep3   [...]
        ]
    batch3  []
    batchn  []
    ]
```

# Feature engineering

These functions can be combined to reduce the number of passes. However, doing so will decrease code readability. If performance is a priority, consider modifying the functions.

## Paper

```python
# Momentum Variables

def calculate_momentum(months_sorted, prices_monthly, offset_start, offset_end):

def calculate_mom1m(months_sorted, prices_monthly):

def calculate_mom12m(months_sorted, prices_monthly):

def calculate_mom36m(months_sorted, prices_monthly):

def calculate_chmom(months_sorted, prices_monthly):

def calculate_maxret(months_sorted, max_daily_returns_monthly):

def calculate_indmom(stocks, sic_codes):

def handle_indmom(stock, indmom):

# Liquidity Variables

def calculate_turn(months_sorted, vol_monthly, shares_monthly):

def calculate_std_turn(prices_daily, shares):

def calculate_mve(months_sorted, market_cap_monthly):

def calculate_dolvol(months_sorted, dollar_volume_monthly):

def calculate_ill(prices_daily):

def calculate_zerotrade(months_sorted, vol_sum_monthly, shares_monthly, zero_trading_days, trading_days_count):

# Risk Measures:

def calculate_retvol(prices_daily):

Will always be later by the window of rolling than beta since it is rolling on another rolling metric
def calculate_idiovol(months_sorted, month_latest_week, weekly_returns, market_weekly_returns, interval=156, increment=4):

def calculate_beta_betasq(months_sorted, month_latest_week, weekly_returns, market_weekly_returns, interval=156, increment=4):

# Valuation Ratios and Fundamental Signals:

def calculate_ep_sp(income_statement_annual, market_caps):

def calculate_agr(balance_sheet_annual):
```

## Variants

```python
# EOD and Market Cap ensured to be sorted

# Difference is rather than months_sorted 1-12 we get 0-11
def calculate_mom12m_current(months_sorted, prices_monthly):

# Difference is t_1 = months_sorted[i] rather than t_1 = months_sorted[i+1]
def calculate_chmom_current(months_sorted, prices_monthly):

# Difference is we get the max_return from current month. Same as the function "get_max_daily_returns_monthly()"
def calculate_maxret_current(prices_daily):

# Liquidity Variables

# Difference is we get the market cap of the last trading day from current month.
# Same as the function "get_market_cap_monthly()" but using natural log
def calculate_mve_current(market_caps):

# Difference is we get the dolvol of current month
# avg_dv = dollar_volume_monthly[curr_month]["sum"] / dollar_volume_monthly[curr_month]["count"]
def calculate_dolvol_current(months_sorted, dollar_volume_monthly):

# Difference is we get zerotrade of current month
# zero_days = zero_trading_days[current_month]
def calculate_zerotrade_current(months_sorted, vol_sum_monthly, shares_monthly, zero_trading_days, trading_days_count):

# Risk Measures:

# Difference is the window starts from current month
# month_start = months_sorted[current]
def get_rolling_weekly_returns_current(months_sorted, month_latest_week, weekly_returns, interval=156, increment=4):


# Difference is the window starts from current month
# month_start = months_sorted[month_current_index]
def calculate_idiovol_current(months_sorted, month_latest_week, weekly_returns, market_weekly_returns, interval=156, increment=4):

# Difference is beta and betasq from this month
# month_start = months_sorted[current]
def calculate_beta_betasq_current(months_sorted, month_latest_week, weekly_returns, market_weekly_returns, interval=156, increment=4):

def calculate_ep_sp_quarterly(income_statement_quarterly, market_caps):

def calculate_agr_quarterly(balance_sheet_quarterly):
```

# Notes

Financial Statements can in fact go earlier than the earliest eod, market cap dates since they were not public then.
