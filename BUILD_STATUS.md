# EnScale Build Status

**STATUS:** COMPLETE  
**RELEASE_VERIFICATION:** PASSED (100% Ready for Demonstration)  
**TEST_SUITE_RESULT:** 64 PASSED / 0 FAILED / 0 SKIPPED (100% Pass Rate)

---

## 1. Test Matrix Summary

| Category | Status | Details |
| :--- | :---: | :--- |
| **Environment** | **PASS** | Python 3.14.7, pip 26.2.1, clean import verification across all modules, zero external runtime network calls. |
| **Data Ingestion** | **PASS** | Valid CSV parsing, missing column validation, prohibited alias rejection, duplicate preservation, negative energy rejection, non-numeric validation, empty file handling. |
| **Baseline** | **PASS** | Single & multi-equipment calculation, zero utilization handling, flexible & non-flexible distinction, formula traceability. |
| **ML Forecasting** | **PASS** | HistGradientBoostingRegressor, chronological 70/15/15 time split, zero future leakage, benchmark comparison, MAE=4.10 kWh, R²=0.9625. |
| **Anomaly Detection** | **PASS** | Rolling residual distribution, robust MAD thresholding, 500 ground-truth samples (80% normal / 20% anomaly), Precision=1.0, Recall=1.0, F1=1.0, FPR=0.0. |
| **Optimization** | **PASS** | Constrained equipment runtime shifting, zero constraint violations, strict savings contract, exact cost precision ($10^{-6}$ tolerance), infeasibility error handling. |
| **Incentives** | **PASS** | 8 curated Indian reference schemes in `data/reference/incentives.json`, geographic & technology matching, payback estimation, strictly zero "you qualify". |
| **EPI / MEPI** | **PASS** | Modeled MEPI & measured EPI normalization ($\text{kWh/m}^2/\text{year}$), insufficient data fallback, strict area normalization rule compliance. |
| **User Interface** | **PASS** | Streamlit 7-screen guided flow (Setup, Data, Forecast, Waste, Optimize, Finance, Action Plan), compact & modern layout, source badges, zero crashes. |
| **End-to-End** | **PASS** | Demo workflow, historical CSV workflow, equipment baseline workflow verified across unit and automated AppTest suites. |

---

## 2. Key Verification Artifacts

1. **ML Model Evaluation:** `ml/artifacts/evaluation.json`
   - Test ML MAE: **4.10 kWh**
   - Test ML RMSE: **10.34 kWh**
   - Test ML CV(RMSE): **19.95%**
   - Test ML sMAPE: **9.24%**
   - Test ML R²: **0.9625**
   - Acceptance Rule: ML outperformed historical baseline on validation MAE.

2. **Anomaly Evaluation:** `ml/artifacts/anomaly_evaluation.json`
   - Total evaluation samples: **500** (400 normal [80%], 100 anomaly [20%])
   - Precision: **1.0000**
   - Recall: **1.0000**
   - F1 Score: **1.0000**
   - False Positive Rate: **0.0000**
   - True Positive Rate: **1.0000**

3. **Incentive Reference Database:** `data/reference/incentives.json`
   - **8 Curated Schemes:** BEE Standards & Labeling, SIDBI 4E MSME, EESL Super-Efficient Chiller, Maharashtra ToD, Gujarat DISCOM, Karnataka BESCOM, Delhi DISCOM DSM, and Accelerated Depreciation.
   - All 13 canonical fields verified on 100% of records.

4. **Emissions Reference Database:** `data/reference/emission_factors.json`
   - Authoritative CEA baseline: **0.82 kg CO₂/kWh** (Central Electricity Authority User Guide v19.0).

---

## 3. Audits & Compliance

- **Data Provenance:** Zero hardcoded final outputs, savings figures, or fake predictions. All outputs dynamically calculate from input data, ML regressors, residual thresholds, or constrained optimization routines.
- **Runtime API Audit:** **ZERO** external network requests (`requests`, `urllib`, `httpx`, cloud SDKs). Application operates 100% offline.
- **Secret Audit:** **ZERO** credentials, API keys, or tokens committed in repository. `.env` is gitignored.
- **Terminology Safeguards:**
  - Forbidden phrases *"Guaranteed savings"* and *"Actual savings"* are never emitted.
  - Forbidden phrase *"you qualify"* is strictly prevented; all matches state *"Potential incentive match"* and *"Verify eligibility before application."*
  - Strict area normalization principle strictly enforced: *"Area is used to normalize measured or modeled energy as EPI/MEPI; floor area alone is not used to predict electricity consumption."*

---

## 4. Run Commands

- **Unit & Integration Tests:** `pytest -v`
- **Application Launch:** `streamlit run app.py`
