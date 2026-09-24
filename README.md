# EnScale

> **"EnScale is an ML-assisted energy decision platform for commercial and small industrial buildings."**

EnScale provides intelligent, transparent, and constrained energy decision support. It empowers facility managers, commercial building operators, and small industrial plants to establish accurate energy baselines, forecast electrical loads, detect operational waste, optimize equipment scheduling, and match applicable efficiency incentives.

---

## 1. Core Pipeline

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

## 2. Product Boundary

### EnScale IS:
- Commercial-building energy decision support
- Small-industrial energy decision support
- Energy forecasting (machine learning regression)
- Energy anomaly and operational waste detection
- Equipment-level energy estimation and baseline modeling
- Constrained equipment runtime scheduling
- Electricity cost optimization
- Energy Performance Index (EPI / MEPI) normalization ($\text{kWh/m}^2/\text{year}$)
- Incentive matching against curated reference efficiency schemes
- Transparent modeled-savings estimation

### EnScale IS NOT:
- IoT hardware
- Live sensor telemetry platform
- Solar design software
- Workforce management system
- Government eligibility authority
- Installer marketplace
- Generic chatbot
- Homeowner solar calculator
- Live utility billing API platform

---

## 3. Local-First Architecture

EnScale is architected to execute **100% locally and offline**. Normal application execution does not require an active internet connection, cloud services, external APIs, authentication microservices, or external database servers.

- **App Framework & UI:** Streamlit
- **Data Processing & Analytics:** pandas, NumPy
- **Machine Learning & Modeling:** scikit-learn, joblib
- **Visualization:** Plotly
- **Test Framework:** pytest (64 unit, integration, and UI tests)
- **Reference Databases:** Curated local JSON structures (`data/reference/`)

---

## 4. Data Ingestion & Operating Modes

EnScale supports exactly three operating modes:

### Mode 1: User Historical Data
- Ingests user-supplied time-series consumption CSVs (e.g., 15-minute or hourly interval meter data).
- **Required Columns:** `timestamp`, `energy_kwh`
- **Optional Columns:** `temperature_c`, `occupancy`, `equipment_load_kw`
- Performs automated timestamp parsing, chronological sorting, duplicate detection, missing-value inspection, negative-energy validation, numeric validity verification, and sampling-frequency inspection.
- Bad data is never silently discarded; structured validation reports are attached.

### Mode 2: Equipment-Based Baseline
- Designed for facilities without interval submetering or historical data.
- Computes baseline using the canonical equipment formula:
  $$\text{estimated\_energy\_kwh} = \text{rated\_power\_kw} \times \text{quantity} \times \text{hours\_per\_day} \times \text{operating\_days} \times \text{utilization\_factor}$$
- Every result remains traceable to its equipment inputs and operating assumptions.

### Mode 3: Demo Scenario
- Out-of-the-box reference scenario: a standard 2,500 m² commercial office building located in Mumbai, Maharashtra, operating Monday–Saturday (08:00–18:00).
- Includes 12 days of hourly interval readings with pre-injected real-world operational anomalies (after-hours HVAC, overnight lighting, extended compressor runtime).

---

## 5. Canonical CSV Format & Downloadable Template

EnScale provides a downloadable CSV template directly from the interface:

```csv
timestamp,energy_kwh,temperature_c,occupancy,equipment_load_kw
2026-09-01 00:00:00,45.2,26.5,5,42.0
2026-09-01 01:00:00,41.8,25.8,2,39.5
2026-09-01 02:00:00,38.4,25.2,0,36.0
```

Prohibited aliases (such as `area`, `sqft`, `energy`, `power`, or `cost`) are strictly rejected by the ingestion validator to prevent schema drift.

---

## 6. Machine Learning Forecasting Methodology

- **Model Architecture:** `HistGradientBoostingRegressor` (Histogram-based gradient boosting regressor with L2 regularization).
- **Feature Engineering:** Cyclic temporal features (`hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`, `month`), outdoor air temperature, and operational indicators. Future leakage is strictly prevented.
- **Data Splitting:** Strict chronological time split into **70% Training**, **15% Validation**, and **15% Test** sets without shuffling.
- **Benchmark Evaluation:** Evaluated against a `SameHourHistoricalBaseline` benchmark model.
- **Acceptance Rule:** The ML regressor must outperform the historical baseline on validation MAE before deployment.
- **Actual Test Set Performance (`ml/artifacts/evaluation.json`):**
  - **MAE:** 4.10 kWh
  - **RMSE:** 10.34 kWh
  - **CV(RMSE):** 19.95%
  - **sMAPE:** 9.24%
  - **R² Score:** 0.9625

---

## 7. Anomaly & Operational Waste Detection

EnScale utilizes a transparent, residual-based anomaly detection framework:

$$\text{forecast\_error\_kwh} = \text{actual\_energy\_kwh} - \text{predicted\_energy\_kwh}$$
$$\text{forecast\_error\_pct} = \left(\frac{|\text{forecast\_error\_kwh}|}{\text{actual\_energy\_kwh}}\right) \times 100$$

- **Statistical Thresholding:** Uses rolling residual distributions with Median and Median Absolute Deviation (MAD) scale estimation ($Z_{\text{robust}} = \frac{e - \text{median}}{1.4826 \times \text{MAD}}$) to remain robust against baseline contamination.
- **Severity Classification:** `normal` ($|Z| < 2.0$), `low` ($2.0 \le |Z| < 2.5$), `medium` ($2.5 \le |Z| < 3.5$), and `high` ($|Z| \ge 3.5$).
- **Quantification:**
  $$\text{waste\_kwh} = \max(0, \text{actual\_energy\_kwh} - \text{predicted\_energy\_kwh})$$
  $$\text{potential\_waste\_cost\_inr} = \text{waste\_kwh} \times \text{tariff\_inr\_per\_kwh}$$
- **Independent Evaluation (`ml/artifacts/anomaly_evaluation.json`):**
  - Evaluated on 500 controlled samples: 400 normal (80%) and 100 controlled anomalies (20%) across after-hours HVAC, overnight lighting, extended compressor runtime, and load spikes.
  - **Precision:** 1.0000 | **Recall:** 1.0000 | **F1 Score:** 1.0000 | **FPR:** 0.0000

---

## 8. Constrained Equipment Optimization & Modeled Savings

EnScale applies deterministic constrained optimization over equipment operating schedules:

### Constraints Enforced:
1. **Facility Operational Window:** Equipment cannot operate outside building operating hours (`operating_start` to `operating_end`).
2. **Equipment Boundaries:** Flexible equipment runtime must satisfy $\text{minimum\_hours} \le h \le \text{maximum\_hours}$.
3. **Non-Flexible Asset Preservation:** Non-flexible equipment is never altered.
4. **Feasibility Checks:** If constraints are physically impossible, raises `InfeasibleConstraintError` with an explicit explanation.

### Canonical Savings Contract:
$$\text{energy\_savings\_kwh} = \text{baseline\_energy\_kwh} - \text{optimized\_energy\_kwh}$$
$$\text{savings\_pct} = \left(\frac{\text{energy\_savings\_kwh}}{\text{baseline\_energy\_kwh}}\right) \times 100$$
$$\text{cost\_savings\_inr} = \text{baseline\_cost\_inr} - \text{optimized\_cost\_inr}$$

> **Important:** "Modeled savings are estimates under the stated operating assumptions and constraints."

The phrases *"Guaranteed savings"* and *"Actual savings"* are strictly forbidden and never emitted. If constraints yield non-positive savings, EnScale outputs *"No modeled savings under the current constraints."* without recommending negative savings.

---

## 9. Environmental Impact & EPI / MEPI Normalization

- **Emissions Reference (`data/reference/emission_factors.json`):**
  - Built on Central Electricity Authority (CEA) User Guide Version 19.0 (0.82 kg CO₂/kWh for the Indian national electricity grid).
  - Explicit provenance statements accompany all carbon calculations.
- **Modeled Energy Performance Index (MEPI):**
  $$\text{MEPI} = \frac{\text{modeled\_annual\_energy\_kwh}}{\text{net\_built\_up\_area\_m2}} \quad (\text{kWh/m}^2/\text{year})$$
- **Measured Energy Performance Index (EPI):**
  $$\text{EPI} = \frac{\text{measured\_annual\_energy\_kwh}}{\text{net\_built\_up\_area\_m2}} \quad (\text{kWh/m}^2/\text{year})$$
  - If a verified 12-month interval dataset is not present, EnScale states: *"Insufficient data to calculate measured EPI."*

> **Critical Area Normalization Principle:**
> "Area is used to normalize measured or modeled energy as EPI/MEPI; floor area alone is not used to predict electricity consumption."

---

## 10. Incentive Policy Matching

EnScale includes an authoritative reference database (`data/reference/incentives.json`) of documented Indian energy efficiency programs:
1. **BEE Standards & Labeling Star-Rated Equipment Program** (National, Motors, Pumps, HVAC)
2. **SIDBI 4E (End-to-End Energy Efficiency) Scheme for MSMEs** (National, Motors, Compressors, Boilers)
3. **EESL Super-Efficient Chiller and Air Conditioning Program** (National, Commercial Cooling)
4. **Maharashtra Time-of-Day (ToD) Off-Peak Energy Tariff Incentive** (Maharashtra, MERC / MSEDCL)
5. **Gujarat DISCOM Industrial & SME Energy Efficiency Tariff Incentive** (Gujarat, GERC / GUVNL)
6. **Karnataka BESCOM Time-of-Day Off-Peak Tariff Concession** (Karnataka, KERC / BESCOM)
7. **Delhi DISCOM Commercial Demand Side Management Rebate Scheme** (Delhi, DERC / BSES / TPDDL)
8. **Commercial Accelerated Depreciation for Clean Energy Assets** (National, MNRE / IT Act Sec 32)

### Compliance & Terminology Safeguards:
- The phrase *"you qualify"* is **strictly prohibited**.
- All matches return: *"Potential incentive match"* and *"Verify eligibility before application."*
- Modeled equipment upgrade payback is computed transparently:
  $$\text{modeled\_payback\_years} = \frac{\text{effective\_modeled\_investment\_inr}}{\text{annual\_modeled\_savings\_inr}}$$
  - If capacity is unmeasured or savings are non-positive, returns *"Payback unavailable."*

---

## 11. Limitations & Analytical Assumptions

1. **Deterministic Decision Support:** EnScale provides modeled decision support; it does not replace professional on-site ASHRAE Level III energy audits.
2. **Local Grid Factors:** Carbon calculations rely on CEA published grid averages (0.82 kg CO₂/kWh) unless overridden with verified regional or captive solar factors.
3. **Submetering Requirement:** Detailed equipment-level anomaly attribution requires interval submeter data (`equipment_load_kw`). When absent, anomalies are labeled generally as *"Consumption anomaly"*.
4. **Regulatory Non-Authority:** EnScale matches technical alignment against documented policy criteria; DISCOMs and nodal agencies retain sole legal authority for rebate sanctioning.

---

## 12. Interactive Demo Walkthrough (11 Steps)

Follow this sequence to test or present EnScale:

1. **Launch App:** Run `streamlit run app.py` and open the local URL in your browser.
2. **Setup Screen:** Review the default facility configuration: 2,500 m² Commercial Office in Mumbai, operating 08:00–18:00 at ₹ 8.50/kWh with an initial inventory of 4 equipment assets (Chiller, Lighting, Air Compressor, IT Servers).
3. **Data Screen:** Select **Demo Scenario**. Notice the purple badge `DATA SOURCE: DEMO DATA` appear.
4. **Inspect Validation:** View the dataset summary (288 hourly intervals, 12-day date span, 0 duplicates, 0 negative energy).
5. **Forecast Screen:** Examine the interactive Plotly Actual vs. Expected load curve. Inspect the verified ML model test accuracy metrics (`MAE: 4.10 kWh`, `R²: 0.9625`, `CV(RMSE): 19.95%`).
6. **Waste Screen:** Identify the detected operational waste intervals. Note the breakdown of actionable waste kWh and associated financial cost (₹).
7. **Optimize Screen:** Run the constrained equipment optimization engine. Review the baseline vs. optimized runtime comparison.
8. **Evaluate Savings:** Verify that flexible runtimes are curtailed while non-flexible equipment is preserved. Note the modeled energy savings (kWh) and cost savings (₹).
9. **Finance Screen:** Review the matched incentive programs (e.g., BEE Star-Rated Equipment, Maharashtra ToD Off-Peak Incentive, EESL Chiller Program) and inspect the modeled equipment upgrade payback.
10. **Action Plan:** Open the executive **ENERGY ACTION PLAN** consolidating the 8 sections (Current Energy Position, Main Detected Waste, Recommended Operational Changes, Estimated Modeled Savings, Potential Upgrades, Matched Incentives, Environmental Impact, Assumptions).
11. **Verification:** Confirm smooth navigation and zero exceptions across the entire workflow.

---

## 13. Installation & Verification

### Prerequisites
- Python 3.10+ (tested on Python 3.12 and Python 3.14)
- Virtual environment (`venv`)

### Clean Installation
```bash
# Clone or navigate to the directory
cd /home/josu/Dev/Projects/EnScale

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Test Suite
```bash
# Run quiet test summary
pytest -q

# Run full verbose test matrix (64 tests)
pytest -v
```

### Running Application
```bash
streamlit run app.py
```

---

## 14. Project Status

- **Status:** **COMPLETE & VERIFIED**
- **Test Matrix Status:** 64 passed, 0 failed, 0 skipped (100% pass rate).
- **Presentation Readiness:** Verified for live demonstration.
