# EnScale Backend and AI Integration Audit

**Audit date:** 2026-09-25  
**Scope:** Current ZIP-delivered repository, inspected before implementation changes.

## Repository state and test baseline

- The current repository contains a five-screen Streamlit workflow in `app.py`, canonical building/equipment/energy/forecast/optimization/impact/opportunity models in `domain/`, deterministic services in `services/`, and a scikit-learn forecasting model in `ml/` with committed joblib artifacts.
- Demo and reference data are in `data/demo/` and `data/reference/`; data setup, model training/evaluation, anomaly evaluation, and headless demo scripts are under `scripts/`.
- Tests cover ingestion, baseline, forecast, anomaly detection, optimization, impact, incentives, opportunities/decision loop, UI, and end-to-end flows.
- Independent baseline command: `pytest -q` → **80 passed, 1 warning, 0 failed** in 4.03s. The warning is pandas date parsing in `services/ingestion_service.py` for an intentionally invalid timestamp test. `BUILD_STATUS.md` and `README.md` are stale at 78 tests.

## Current architecture and calculation ownership

`app.py` owns Streamlit navigation/session state and also invokes backend calculations directly. Screen 3 loads the cached `EnergyPredictor`, predicts usage, calls anomaly detection, explains anomaly episodes, finds opportunities, and stores outputs. Screen 4 directly calls constrained equipment schedule optimization and impact calculation. Screen 5 may repeat optimization/impact/opportunity/incentive calculations and renders the decision summary, opportunity cards, upgrade illustration, policy matches, environmental impact, downloadable text plan, and technical appendix.

The calculations are deterministic except the trained forecasting model's ML inference. `services/baseline_service.py` calculates equipment estimates and coverage; `services/validation_service.py` validates the building, equipment, energy data, and incentives; `services/anomaly_service.py` computes forecast residuals and statistical anomaly flags; `services/opportunity_service.py` turns episodes, equipment constraints, and residuals into typed opportunities and a decision summary; `services/optimization_service.py` computes constrained schedule recommendations; `services/impact_service.py` calculates modeled kWh/cost/CO2 impacts; `services/incentive_service.py` matches bundled reference programs and computes upgrade payback estimates. Ingestion and reference lookups are local.

## AI status

- The product currently describes itself as **ML-assisted**. Forecasting is ML-based (`HistGradientBoostingRegressor` via `ml/predict.py`); anomaly detection is statistical/rule-based; opportunity ranking/explanation, optimization, impact, incentive matching, and report generation are deterministic.
- No LLM provider, provider abstraction, model API client, or `AI_PROVIDER` configuration exists in the inspected source. No OpenAI, Gemini, or Ollama integration was found.
- Streamlit currently calls ML and deterministic backend services directly; there is no structured AI analysis context or analysis progress contract.

## Recommendation and report pipeline

The current recommendation inputs are the building profile, equipment inventory and its operating constraints, energy dataframe, ML forecast, actual-vs-expected residuals, anomaly dataframe/episodes, tariff, emission factor, optimizer result, impact result, and local incentive matches. `identify_energy_opportunities()` builds schedule-derived and model/rule-based `EnergyOpportunity` records. `generate_decision_summary()` reconciles schedule-derived headline savings separately from model-flagged historic excess and returns display strings plus numeric summary values.

The current Action Plan is assembled/rendered in Screen 5 of `app.py`, not by a dedicated report service. It uses the decision summary, typed opportunities, optimization and impact results, incentive matches, and explanatory static copy. A text download is generated in the UI. Some report copy is illustrative/static (including a fixed upgrade payback/range); the summary also substitutes hard-coded fallback values when anomaly data or incentive matches are absent. These are material grounding risks for any structured AI/report integration and must not be passed as measured facts.

## Risks to integration

1. `app.py` currently combines UI state, orchestration, and calculations; the requested ownership boundary argues for a backend pipeline that can be called without changing the Streamlit UI.
2. `generate_decision_summary()` has hard-coded no-data anomaly and no-match incentive fallbacks. These must be distinguished from actual analysis facts before any AI interpretation or backend report contract uses them.
3. Existing opportunity objects have rich evidence/traceability fields but do not expose the exact proposed generic UI contract names; adapters can preserve existing calculations while normalizing output.
4. Action Plan content includes static upgrade claims, so new structured report output should derive facts from current results and label limitations rather than treat static display claims as calculated evidence.
5. There is no `.env.example` or AI dependency today. Optional cloud/local provider dependencies and credentials must remain isolated and defaults offline/deterministic.
6. `BUILD_STATUS.md`/`README.md` test totals are stale. `.gitignore` ignores `.env` and runtime caches but should explicitly cover local AI/model caches and secrets as needed.

## Files likely to change

- New provider layer: `services/ai/{__init__.py,base.py,schemas.py,deterministic_provider.py,ollama_provider.py,gemini_provider.py,openai_provider.py,manager.py}`.
- New backend orchestration/contracts: analysis pipeline and progress/report structures under `services/` (new modules), plus `BACKEND_UI_CONTRACT.md`.
- New provider diagnostic: `scripts/check_ai.py`.
- Configuration/documentation: `config/settings.py`, `requirements.txt`, `.env.example`, `.gitignore`, `README.md`, `AI_SETUP.md`, and `BUILD_STATUS.md`.
- New targeted backend tests under `tests/`.
- `app.py` is not planned for modification unless integration testing proves a small compatibility adapter is necessary.

No implementation changes were made before this audit was written.
