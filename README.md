# OceanClean 🌊

Reproducible **offline gap reconstruction** for ocean buoy time series.

**Status:** initial data pipeline and baseline benchmark. No claimed research novelty,
no trained deep model. First real-data offline reconstruction results are available below.

Can weather observations help reconstruct missing sea-water temperature? When do
long outages make simple interpolation unreliable? This repository establishes a
controlled benchmark before adding neighboring buoys and learned reliability weights.

## First real-data results

[41009 / 2024 experiment report](reports/first_experiment/README.md): 52,692 raw
observations, two outage scenarios, three seeds. Linear interpolation outperformed
the initial learned baseline. See the report for training-budget and evaluation limits.

## Quick start

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
ocean-clean download --station 41009 --year 2024 --source thredds
ocean-clean quality --csv data/raw/41009h2024.csv
ocean-clean benchmark --csv data/raw/41009h2024.csv --scenario sensor
ocean-clean benchmark --csv data/raw/41009h2024.csv --scenario station --out outputs/station
python -m unittest discover -s tests -v
```

41009/2024 was downloaded and audited for the first pilot. Download needs
network access to NOAA. A failed download raises an error; no synthetic data is
substituted. Raw archives are cached with their source URL and SHA-256 checksum.

## What is implemented

- NDBC standard meteorological text/gzip parsing, including legacy year/header forms.
- Column-specific missing-value handling (99 degrees is a valid wind direction).
- UTC hourly alignment and per-variable missingness/longest-gap reports.
- Chronological 60/20/20 train/validation/test partitions.
- Reproducible 1/6/24/72-hour masks, generated separately within each partition.
- Linear interpolation, training-median, and gradient-boosting baselines.
- Sensor-only outages versus all-column station outages.
- MAE/RMSE by split, gap length, and method; point-level predictions for inspection.

## Evaluation contract

This is **retrospective reconstruction**, not forecasting: observations after a gap
are allowed. Target values are masked before feature construction. Training labels
come only from artificially hidden training observations. Validation and test labels
are used only for scoring. Baseline hyperparameters are fixed; test results must not
be used for tuning. Each gap length trains a separate baseline model.

Blocks have observed endpoints and no original target missingness inside them.
Thus this benchmark excludes difficult edge gaps and cannot establish accuracy on
real missing data. Sampling may yield fewer blocks than requested; actual counts
are reported. Archive observations are references, not perfect physical truth.

The first model uses contemporaneous non-target scalar variables and a masked
interpolation feature. Station outages remove these covariates too. Neighboring
stations, predictive intervals, event preservation metrics, and neural networks are
**not implemented yet**. Wind/wave directions are excluded before hourly averaging
because circular quantities require different aggregation.

## Data and sources

- [NOAA NDBC historical archive](https://www.ndbc.noaa.gov/historical_data.shtml)
- [Variable definitions, units, and missing codes](https://www.ndbc.noaa.gov/faq/measdes.shtml)
- [Archive processing](https://dods.ndbc.noaa.gov/)

Archive quality control may remove real failures. Synthetic corruption performance
must not be presented as real sensor-fault detection performance. Keep downloaded
data and generated output outside git; retain manifests for reproducibility.

## Roadmap

1. Audit 5–10 candidate stations and overlapping years; freeze a dataset manifest.
2. Expand training corruption examples and add multi-seed uncertainty estimates.
3. Add neighboring-buoy features with a station-dropout ablation.
4. Evaluate a temporal neural model against the frozen baselines.
5. Add an inspection UI and uncertainty calibration on validation data.

## Layout

`src/ocean_clean/data.py`: ingest and quality audit. `benchmark.py`: masks, features,
training and scoring. `cli.py`: command-line entry points. `tests/`: leakage,
missing-code, temporal-split and end-to-end smoke checks.
