# EnScale Build Status

**STATUS:** HEALTHY (Awaiting Kaggle Credentials / Dataset Files for ASHRAE Download)
**CURRENT_STAGE:** Stage 2 - Data Ingestion, Equipment Baseline, Demo Data, and ML Forecasting Foundation

## COMPLETED:
- **Verified Stage 1 Foundation**: Re-tested `pytest` (24/24 passing) and Streamlit headless execution prior to Stage 2 development.
- **Historical CSV Ingestion (`services/ingestion_service.py`)**:
  - Implemented `load_energy_csv(file) -> pandas.DataFrame`.
  - Enforced canonical column requirements (`timestamp`, `energy_kwh`).
  - Implemented timestamp parsing, chronological sorting, duplicate detection, missing-value detection, negative-energy validation, numeric checks, and sampling-frequency inspection.
  - Attached non-destructive structured validation details to `df.attrs["validation_info"]`.
- **Downloadable CSV Template (`data/reference/energy_template.csv`)**:
  - Created canonical template with columns: `timestamp,energy_kwh,temperature_c,occupancy,equipment_load_kw`.
  - Added UI download button integration via `get_energy_csv_template()`.
- **Equipment Baseline (`services/baseline_service.py`)**:
  - Implemented `calculate_equipment_energy(equipment_list)` using canonical formula:
    $$\text{estimated\_energy\_kwh} = \text{rated\_power\_kw} \times \text{quantity} \times \text{hours\_per\_day} \times \text{operating\_days} \times \text{utilization\_factor}$$
  - Returns per-equipment energy with explicit arithmetic calculation trace, total energy, and transparent assumptions.
- **Deterministic Demo Data (`scripts/setup_data.py`)**:
  - Generated 1,488 rows of hourly simulated office data (seed=42) in `data/demo/demo_office_energy.csv`.
  - Labeled prominently: `"Demo dataset — simulated"`.
  - Injected controlled operational waste anomalies (HVAC after hours, lighting overnight, compressor extended operation) documented in `data/demo/demo_manifest.json`.
- **ML Pipeline & Evaluation Modules**:
  - `ml/preprocessing.py`: Generates `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `air_temperature`, etc.
  - Chronological split: Exactly 70% train, 15% validation, 15% test with no shuffling and strict no-leakage verification.
  - `ml/evaluation.py`: Implemented non-ML `SameHourHistoricalBaseline`, calculated canonical metrics (`MAE`, `RMSE`, `CV(RMSE)`, `sMAPE`, `R²`), and computed `improvement_vs_baseline_pct`.
  - `ml/train.py`: Orchestrates training using `HistGradientBoostingRegressor`, tests against baseline on validation MAE, evaluates untouched test set, and exports `model.joblib`, `metadata.json`, and `evaluation.json`.
  - `ml/predict.py`: Performs inference on new dataframes with fallback defaults for missing optional features.
- **Streamlit UI Integration (`app.py`)**:
  - Implemented interactive Data screen supporting 3 selectable modes:
    1. `Use My Historical Data`: CSV uploader, downloadable template, validation metrics card, advisories, and data preview.
    2. `Build Baseline From Equipment`: Interactive equipment inventory table, total modeled baseline, and traceable per-equipment breakdown.
    3. `Demo Scenario`: Simulated demo dataset display, anomaly explanations, and time-series preview.
- **ASHRAE Processing Pipeline Preparation (`scripts/process_ashrae.py`)**:
  - Built deterministic subset processor for `meter == 0` (electricity) and building IDs `[0, 1, 2, 3, 4]` merged with weather and metadata, ready to generate `data/reference/dataset_manifest.json`.

## VERIFIED:
- `pytest` executed with 32 passing tests out of 32 (100% pass rate, 0 failures, 0 warnings).
- `streamlit run app.py --server.headless=true` verified running locally on port 8501 without errors.
- Verified absence of future target leakage in feature generation and chronological splitting.
- Verified calculation traceability for equipment energy calculations.

## KNOWN_ISSUES:
- **Kaggle Credentials Unavailable**: As instructed by Section 6 ("If Kaggle credentials or download access are required and unavailable: STOP and report the exact issue. Do NOT silently use another dataset."), downloading the raw ASHRAE Great Energy Predictor III dataset directly from Kaggle requires an authenticated Kaggle account that has accepted the competition rules. No `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables exist in the environment, and unauthenticated requests return `HTTP 401 Unauthorized`. Development was stopped at this boundary without silently substituting another dataset.

## NEXT_STAGE:
- Obtain Kaggle credentials or place ASHRAE raw files (`train.csv`, `building_metadata.csv`, `weather_train.csv`) into `data/raw/` to run `scripts/process_ashrae.py` and train the ML model on the real ASHRAE dataset.
