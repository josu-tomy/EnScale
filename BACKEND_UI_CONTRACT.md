# Backend to UI Contract

The backend pipeline is called with `analyze_energy(profile, equipment, dataframe, predictor=None, ai_manager=None, emission_factor_kg_per_kwh=0.82)` from `services.analysis_service`. Inputs use existing `BuildingProfile`, `Equipment`, and validated canonical energy columns. The UI can call it and retain `AnalysisResult.to_dict()` in its session state. The current Streamlit app remains compatible and does not yet call this new contract.

## Session state

Recommended keys: `building_profile`, `equipment_inventory`, `active_df`, `analysis_result`, `analysis_progress`, and `analysis_error`. The result is a plain-serializable dictionary; domain calculations are kept in backend services.

## Analysis result

`AnalysisResult` includes `analysis_context`, `forecast`, `anomalies`, `opportunities`, `optimization`, `impact`, `incentives`, `ai_insight`, `ai_provider`, `action_plan`, and `progress`. All numerical fields in these sections come from deterministic engine output. AI text is interpretive and the deterministic facts remain separate.

## Progress

`progress` is an ordered list of `{key, label, status, message}` entries for `VALIDATING_DATA`, `BUILDING_BASELINE`, `FORECASTING_USAGE`, `CHECKING_ANOMALIES`, `FINDING_OPPORTUNITIES`, `OPTIMIZING_ACTIONS`, `CALCULATING_IMPACT`, `MATCHING_INCENTIVES`, `GENERATING_AI_INSIGHT`, and `FINALIZING_ACTION_PLAN`. Status is `pending`, `running`, `complete`, `failed`, or `skipped`. A synchronous call reports actual completed work at return or raises with `exception.analysis_progress` after marking the active stage failed. Stages are not marked complete before their work returns.

## AI provider state

`ai_provider` contains `provider`, `configured`, `reachable`, `model_available`, `generation_successful`, `fallback_active`, and a credential-safe `message`. Provider failure uses local deterministic reasoning only. No cloud-to-cloud switching occurs. Provider selection is `AI_PROVIDER=deterministic|ollama|gemini|openai`.

## Action Plan/report

`action_plan` contains `executive_summary`, `current_energy_state`, `key_finding`, `main_opportunity`, `evidence`, `recommended_action`, `expected_impact`, `implementation_effort`, `equipment_to_review`, `upgrade_opportunity`, `incentive_matches`, `confidence`, `limitations`, and `technical_appendix`. Unsupported effort/upgrade ratings are nullable or explanatory text. Empty opportunities use `main_opportunity: null` and the deterministic no-significant-opportunity statement. Empty incentives are an empty list, not a fabricated match.

## Errors and empty states

Invalid building, equipment, or energy input raises `ValueError`; the active progress stage is `failed`. The UI should show the validation message and retain inputs for correction. Empty input should be rejected by validation. A valid dataset with no detected opportunities is a successful result and displays “No significant efficiency opportunity was detected from the available data.” No incentive matches is a valid empty list. Provider errors do not fail the analysis; the fallback is identified in `ai_provider` and limitations.

## Expected UI behavior

Render deterministic result metrics from analysis result data. Label optimization/impact as modeled. Render provider/fallback status and limitations. Do not use AI prose as a source of numeric values or claim eligibility based on a match. Render progress from returned statuses, and preserve empty lists/nulls as empty states.
