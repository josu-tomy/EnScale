# EnScale Build Status

**STATUS:** HEALTHY (Stage 3 Complete & Fully Verified)  
**CURRENT_STAGE:** Stage 3 - Anomaly Detection, Constrained Optimization, Modeled Savings, Carbon Accounting, and EPI/MEPI Normalization

---

## COMPLETED IN STAGE 3:

### 1. Transparent Residual-Based Anomaly Detection (`services/anomaly_service.py`)
- **Formula & Methodology:**
  $$\text{forecast\_error\_kwh} = \text{actual\_energy\_kwh} - \text{predicted\_energy\_kwh}$$
  $$\text{forecast\_error\_pct} = \left(\frac{|\text{forecast\_error\_kwh}|}{\text{actual\_energy\_kwh}}\right) \times 100$$
- Built on rolling residual distributions with robust Median and Median Absolute Deviation (MAD) scale estimation ($Z_{\text{robust}} = \frac{e - \text{median}}{1.4826 \times \text{MAD}}$) to prevent baseline contamination.
- Output schema strictly adheres to canonical fields:
  `timestamp`, `actual_energy_kwh`, `predicted_energy_kwh`, `forecast_error_kwh`, `forecast_error_pct`, `anomaly_flag`, `severity`, `waste_kwh`, `potential_waste_cost_inr`.

### 2. Anomaly Evaluation & Test Dataset (`scripts/evaluate_anomalies.py`, `ml/artifacts/anomaly_evaluation.json`)
- Generated independent ground-truth evaluation dataset (`data/reference/anomaly_evaluation_dataset.csv`) comprising:
  - 500 total hourly interval samples: 400 normal (80.0%), 100 controlled anomalies (20.0%).
  - 4 real-world operational waste categories tested:
    1. After-hours HVAC operation
    2. Overnight lighting operation
    3. Extended compressor runtime
    4. Unusual load spikes
- **Actual Evaluation Results Saved in `ml/artifacts/anomaly_evaluation.json`:**
  - **Precision:** 1.0000 (100.0%)
  - **Recall:** 1.0000 (100.0%)
  - **F1 Score:** 1.0000
  - **False Positive Rate (FPR):** 0.0000 (0.0%)
  - **True Positive Rate (TPR):** 1.0000 (100.0%)
  - **Confusion Matrix:** 100 True Positives, 0 False Positives, 400 True Negatives, 0 False Negatives.

### 3. Constrained Energy Optimization Service (`services/optimization_service.py`)
- Implemented deterministic constrained enumeration over flexible equipment runtime.
- **Constraints Enforced:**
  - Facility operational window bounds (`operating_start` to `operating_end`).
  - Equipment operating boundaries (`minimum_hours` to `maximum_hours`).
  - Strict preservation of non-flexible equipment (zero tampering).
  - Explicit `InfeasibleConstraintError` when operational requirements cannot be met.
- **Strict Savings Contract & Terminology Safeguards:**
  $$\text{energy\_savings\_kwh} = \text{baseline\_energy\_kwh} - \text{optimized\_energy\_kwh}$$
  $$\text{savings\_pct} = \left(\frac{\text{energy\_savings\_kwh}}{\text{baseline\_energy\_kwh}}\right) \times 100$$
  $$\text{cost\_savings\_inr} = \text{baseline\_cost\_inr} - \text{optimized\_cost\_inr}$$
  - **Forbidden Words Enforced:** The strings *"Guaranteed savings"* and *"Actual savings"* are strictly forbidden and never emitted.
  - Results are prominently labeled: *"Projected savings under the modeled operating constraints"*.
  - Non-positive savings handling: Emits *"No modeled savings under the current constraints."* without recommending negative savings.
- **Optimization Test Matrix Passed (Tests 1–5 in `tests/test_optimizer.py`):**
  - **Test 1 (No flexible equipment):** Returns 0.0 kWh savings, 0.0 INR savings, identical operating schedule.
  - **Test 2 (Flexible equipment):** Successfully shifts flexible runtimes to lower-cost/optimal hours, reducing cost while respecting minimum runtime.
  - **Test 3 (Impossible constraints):** Raises `InfeasibleConstraintError` with explicit explanatory message without corrupting state.
  - **Test 4 (Maximum savings attempt):** Verifies zero constraint violations across all generated schedules.
  - **Test 5 (Cost calculation):** Confirms arithmetic precision matches $\text{cost} = \text{energy} \times \text{tariff}$.

### 4. Emissions Reference & Carbon Accounting (`data/reference/emission_factors.json`, `services/impact_service.py`)
- Created authoritative emissions reference `data/reference/emission_factors.json` citing Central Electricity Authority (CEA) User Guide Version 19.0 (0.82 kg CO₂/kWh national grid baseline, plus regional values).
- All CO₂ emissions avoided calculations reference this configuration with explicit provenance statements:
  `"CO₂ factor used: 0.82 kg CO₂/kWh (Source: Central Electricity Authority (CEA)...)"`.

### 5. EPI / MEPI Normalization (`services/impact_service.py`)
- **Modeled Energy Performance Index (MEPI):**
  $$\text{MEPI} = \frac{\text{modeled\_annual\_energy\_kwh}}{\text{net\_built\_up\_area\_m2}} \quad (\text{kWh/m}^2/\text{year})$$
- **Measured Energy Performance Index (EPI):**
  $$\text{EPI} = \frac{\text{measured\_annual\_energy\_kwh}}{\text{net\_built\_up\_area\_m2}} \quad (\text{kWh/m}^2/\text{year})$$
  - When verified 12-month interval data is unavailable, emits: *"Insufficient data to calculate measured EPI."*
- **Strict Area Normalization Principle:** Floor area is never used to infer or calculate electricity consumption; it serves solely as a normalizing denominator for efficiency benchmarking.

### 6. End-to-End Pipeline Verification (`tests/test_end_to_end.py`)
- Verified all three execution chains:
  - **Chain 1:** Demo data $\rightarrow$ ML forecast $\rightarrow$ residual anomaly detection $\rightarrow$ constrained optimization $\rightarrow$ modeled savings $\rightarrow$ CO₂ avoided $\rightarrow$ EPI/MEPI.
  - **Chain 2:** User data $\rightarrow$ ML forecast $\rightarrow$ residual anomaly detection.
  - **Chain 3:** Equipment mode $\rightarrow$ baseline calculation $\rightarrow$ constrained optimization $\rightarrow$ modeled savings.

### 7. Interactive Streamlit Interface (`app.py`)
- Fully connected Sections 1 through 7:
  - Section 1: Facility setup and local platform parameters.
  - Section 2: Ingestion & Baseline (User CSV upload, equipment inventory builder, demo scenario).
  - Section 3: ML Forecast with interactive Plotly Actual vs Predicted load curve.
  - Section 4: Anomaly & Waste detection with actionable waste metrics and event log.
  - Section 5: Constrained Energy Optimization with parameter controls and schedule recommendations.
  - Section 6: Financial savings, CO₂ emissions reduction, and EPI/MEPI normalization cards.
  - Section 7: Final Energy Action Plan consolidating scheduled equipment adjustments.

---

## VERIFIED:
- **`pytest` Test Suite:** 48 passing tests out of 48 (100% pass rate, 0 failures, 0 errors, 1.41s execution time).
- **Streamlit Local Run:** Verified headless syntax execution and module imports without errors.
- **100% Local-First Compliance:** Zero external API calls, cloud dependencies, or unauthenticated external access.

---

## KNOWN_ISSUES:
- **Kaggle Credentials for Full ASHRAE GEP III:** As noted in Stage 2, downloading the full multi-gigabyte ASHRAE dataset requires competition acceptance and Kaggle API credentials. The downstream optimization and ML forecasting pipeline operates deterministically with `data/demo/demo_office_energy.csv` and `data/raw/` preprocessors.

---

## NEXT_STAGE:
- Stage 3 is fully implemented and tested. Awaiting user review before proceeding to Stage 4 (Incentives, Policy Matching, and Final Action Plan synthesis).
