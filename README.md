# EnScale

> **"EnScale is an ML-assisted energy decision platform for commercial and small industrial buildings."**

EnScale provides intelligent, transparent, and constrained energy decision support. It empowers facility managers, commercial building operators, and small industrial plants to establish accurate energy baselines, forecast loads, detect operational waste, optimize equipment scheduling, and match applicable efficiency incentives.

---

## Core Pipeline

```
USER / BUILDING INPUT
        ↓
DATA INGESTION
        ↓
ENERGY BASELINE
        ↓
ML FORECAST
        ↓
ACTUAL VS EXPECTED
        ↓
ANOMALY / WASTE DETECTION
        ↓
CONSTRAINED ENERGY OPTIMIZATION
        ↓
BEFORE VS AFTER
        ↓
₹ / kWh / CO₂ IMPACT
        ↓
INCENTIVE MATCHING
        ↓
FINAL ENERGY ACTION PLAN
```

---

## Product Boundary

### EnScale IS:
- Commercial-building energy decision support
- Small-industrial energy decision support
- Energy forecasting (ML-driven regression)
- Energy anomaly and waste detection
- Equipment-level energy estimation
- Constrained equipment scheduling
- Electricity-cost optimization
- EPI/MEPI-style energy normalization (kWh/m²/year)
- Incentive matching against reference efficiency programs
- Transparent modeled-savings estimation

### EnScale IS NOT:
- IoT hardware
- Live sensor platform
- Solar design software
- Workforce management
- Government eligibility authority
- Installer marketplace
- Generic chatbot
- Homeowner solar calculator
- Live utility API platform

---

## Local-First Architecture

EnScale is architected to run **100% locally**. Normal application execution does not require an internet connection, cloud services, external APIs, authentication microservices, or external database servers.

- **Frontend & App Interface**: Streamlit
- **Data Processing & Analytics**: pandas, NumPy
- **Machine Learning & Modeling**: scikit-learn, joblib
- **Visualization**: Plotly, matplotlib
- **Test Framework**: pytest
- **Optional Local Persistence**: SQLite (only if local persistence is genuinely required)

### Structured AI interpretation

EnScale's deterministic services remain the source of every numeric result. Optional providers interpret supplied structured findings and fall back to deterministic local reasoning on failure. Select exactly one provider with `AI_PROVIDER=deterministic|ollama|gemini|openai`; cloud providers are never substituted for one another. See [AI_SETUP.md](AI_SETUP.md) and [BACKEND_UI_CONTRACT.md](BACKEND_UI_CONTRACT.md). The backend callable is `services.analysis_service.analyze_energy(...)`; the existing Streamlit UI is not wired to this new contract.

---

## Data Modes

EnScale supports exactly three operating modes:

1. **MODE 1: User Historical Data**
   - Ingests user-supplied time-series consumption CSVs (e.g. 15-minute or hourly interval meter data).
   - Required columns: `timestamp`, `energy_kwh`.
   - Optional columns: `temperature_c`, `occupancy`, `equipment_load_kw`.

2. **MODE 2: Equipment-Based Baseline**
   - For facilities without submetering or historical time-series data.
   - Computes baseline using the canonical equipment formula:
     $$\text{estimated\_energy\_kwh} = \text{rated\_power\_kw} \times \text{quantity} \times \text{hours\_per\_day} \times \text{operating\_days} \times \text{utilization\_factor}$$
   - *Note: Building area alone is NEVER used to calculate energy consumption.*

3. **MODE 3: Demo Scenario**
   - Out-of-the-box reference scenarios (e.g., standard 2,500 m² commercial office in Mumbai) allowing instant exploration without user uploads.

---

## Canonical Variable Contract

To prevent confusion and schema drift, canonical variable names are strictly enforced:

- **Building**: `building_id`, `building_type`, `location_state`, `climate_zone`, `net_built_up_area_m2`, `operating_start`, `operating_end`, `operating_days_per_month`, `occupancy`, `monthly_budget_inr`, `tariff_inr_per_kwh`
- **Equipment**: `equipment_id`, `equipment_type`, `equipment_name`, `rated_power_kw`, `quantity`, `hours_per_day`, `operating_days`, `utilization_factor`, `minimum_hours`, `maximum_hours`, `is_flexible`
- **Energy**: `timestamp`, `energy_kwh`, `temperature_c`, `occupancy`, `equipment_load_kw`
- **Forecast**: `predicted_energy_kwh`, `actual_energy_kwh`, `forecast_error_kwh`, `forecast_error_pct`
- **Optimization (Modeled Savings)**: `baseline_energy_kwh`, `optimized_energy_kwh`, `energy_savings_kwh`, `baseline_cost_inr`, `optimized_cost_inr`, `cost_savings_inr`, `savings_pct`
- **EPI**: `annual_energy_kwh`, `epi_kwh_m2_year`

*Aliases such as `area`, `sqft`, `floor_area`, `energy`, `power`, or `cost` are strictly prohibited.*

---

## Installation & Running

### Prerequisites
- Python 3.10+ (tested with Python 3.12 and 3.14)
- Virtual environment (`venv`)

### Setup
```bash
# Clone or navigate to the directory
cd /path/to/EnScale

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
pytest
```

### Running Streamlit Application
```bash
streamlit run app.py
```

### Generating Demo Data & Training Baseline Model
```bash
python scripts/setup_data.py
python scripts/train_model.py
```

### Model Evaluation Metrics (Benchmark)
Trained with chronological splitting (70% train / 15% validation / 15% test, zero lookahead leakage) on 1,488 hourly intervals:
- **Architecture**: `HistGradientBoostingRegressor` (Scikit-Learn)
- **Test Samples**: 298 samples (untouched chronological test slice)
- **Test $R^2$**: `0.9832`
- **Test MAE**: `3.63 kWh` (3.6345 kWh)
- **Test CV(RMSE)**: `12.89%`
- **Test sMAPE**: `10.05%`

---

## Current Stage & Verification

- **Status**: Complete & Verified (ML-Assisted Energy Decision Loop Engine)
- **Test Suite**: 93 / 93 Passing Tests (`pytest -q`, including AI integration coverage)
- **Runnable Entrypoint**: `app.py` (`streamlit run app.py`)
