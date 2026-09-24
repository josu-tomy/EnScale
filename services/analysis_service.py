"""Backend orchestration and UI-consumable progress/report contracts."""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from services.ai.manager import AIManager
from services.ai.schemas import StructuredAnalysisContext
from services.anomaly_service import detect_anomalies
from services.baseline_service import calculate_equipment_energy
from services.impact_service import calculate_impact
from services.incentive_service import find_incentives
from services.opportunity_service import identify_energy_opportunities
from services.optimization_service import optimize_equipment_schedule
from services.validation_service import validate_building_profile, validate_energy_dataframe, validate_equipment


class AnalysisStage(str, Enum):
    VALIDATING_DATA="VALIDATING_DATA"; BUILDING_BASELINE="BUILDING_BASELINE"; FORECASTING_USAGE="FORECASTING_USAGE"
    CHECKING_ANOMALIES="CHECKING_ANOMALIES"; FINDING_OPPORTUNITIES="FINDING_OPPORTUNITIES"; OPTIMIZING_ACTIONS="OPTIMIZING_ACTIONS"
    CALCULATING_IMPACT="CALCULATING_IMPACT"; MATCHING_INCENTIVES="MATCHING_INCENTIVES"; GENERATING_AI_INSIGHT="GENERATING_AI_INSIGHT"
    FINALIZING_ACTION_PLAN="FINALIZING_ACTION_PLAN"

STAGE_LABELS = {AnalysisStage.VALIDATING_DATA:"Validating energy data", AnalysisStage.BUILDING_BASELINE:"Building energy baseline",
 AnalysisStage.FORECASTING_USAGE:"Forecasting usage", AnalysisStage.CHECKING_ANOMALIES:"Checking anomalies",
 AnalysisStage.FINDING_OPPORTUNITIES:"Finding opportunities", AnalysisStage.OPTIMIZING_ACTIONS:"Optimizing actions",
 AnalysisStage.CALCULATING_IMPACT:"Calculating modeled impact", AnalysisStage.MATCHING_INCENTIVES:"Matching incentives",
 AnalysisStage.GENERATING_AI_INSIGHT:"Generating analysis insight", AnalysisStage.FINALIZING_ACTION_PLAN:"Finalizing action plan"}


@dataclass
class ProgressItem:
    key: str
    label: str
    status: str = "pending"
    message: Optional[str] = None


class AnalysisProgress:
    def __init__(self): self.stages = [ProgressItem(s.value, STAGE_LABELS[s]) for s in AnalysisStage]
    def update(self, key, status, message=None):
        if status not in {"pending", "running", "complete", "failed", "skipped"}: raise ValueError("invalid stage status")
        next(item for item in self.stages if item.key == key).status = status
        next(item for item in self.stages if item.key == key).message = message
    def to_dict(self): return [asdict(s) for s in self.stages]


@dataclass
class AnalysisResult:
    analysis_context: Dict[str, Any]
    forecast: List[float]
    anomalies: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]
    optimization: Dict[str, Any]
    impact: Dict[str, Any]
    incentives: List[Dict[str, Any]]
    ai_insight: Dict[str, Any]
    ai_provider: Dict[str, Any]
    action_plan: Dict[str, Any]
    progress: List[Dict[str, Any]]

    def to_dict(self): return asdict(self)


def _records(df):
    return df.replace({np.nan: None}).to_dict(orient="records")


def build_analysis_context(profile, equipment, df, forecast, anomalies, opportunities, optimization, impact, incentives):
    """Serialize deterministic facts only; no invented report defaults."""
    return StructuredAnalysisContext(
        facts={"building_type":profile.building_type, "location_state":profile.location_state,
               "period_rows":int(len(df)), "actual_energy_kwh":round(float(df.energy_kwh.sum()), 2),
               "forecast_energy_kwh":round(float(np.sum(forecast)), 2),
               "anomaly_count":int(anomalies.anomaly_flag.sum()),
               "optimization":optimization.to_dict(), "impact":impact.to_dict(),
               "incentive_count":len(incentives),
               "equipment_baseline":calculate_equipment_energy(equipment)},
        evidence=[{"title":o.opportunity_name, "reason":o.current_condition,
                   "recommended_action":o.possible_intervention, "category":o.category,
                   "confidence":o.confidence_basis, "metrics":{"annual_avoidable_energy_kwh":o.annual_avoidable_energy_kwh,
                   "estimated_annual_cost_savings_inr":o.estimated_annual_cost_savings_inr}}
                  for o in opportunities],
        limitations=["Modeled impacts depend on user-entered operating constraints.",
                     "Meter data alone may not identify the physical cause of a deviation.",
                     "Incentive matches require verification with the program authority."])


def _action_plan(context, insight, opportunities, impact, incentives, equipment):
    main = opportunities[0] if opportunities else None
    return {"executive_summary":insight.summary,
      "current_energy_state":context.facts,
      "key_finding":insight.key_finding,
      "main_opportunity":main.to_dict() if main else None,
      "evidence":context.evidence,
      "recommended_action":insight.recommendations,
      "expected_impact":impact.to_dict(),
      "implementation_effort":None,
      "equipment_to_review":sorted({e.equipment_name for e in equipment if any(o.equipment_id == e.equipment_id for o in opportunities)}),
      "upgrade_opportunity":insight.upgrade_opportunity,
      "incentive_matches":[m for m in incentives],
      "confidence":insight.confidence,
      "limitations":list(dict.fromkeys(context.limitations + insight.limitations)),
      "technical_appendix":{"analysis_context":context.to_dict()}}


def analyze_energy(profile, equipment, df, predictor=None, ai_manager=None, emission_factor_kg_per_kwh=0.82):
    """Execute actual analysis stages. Exceptions mark the active stage failed and propagate."""
    progress = AnalysisProgress(); current = None
    def begin(stage):
        nonlocal current
        current = stage; progress.update(stage, "running")
    def done(stage): progress.update(stage, "complete")
    try:
        begin(AnalysisStage.VALIDATING_DATA.value)
        valid = validate_building_profile(profile)
        ev = validate_energy_dataframe(df)
        if not valid.is_valid or not ev.is_valid: raise ValueError("; ".join(valid.errors + ev.errors))
        eq_errors = [x for e in equipment if not (x := validate_equipment(e)).is_valid for _ in [0]]
        if eq_errors: raise ValueError("; ".join(err for x in eq_errors for err in x.errors))
        done(current)

        begin(AnalysisStage.BUILDING_BASELINE.value); calculate_equipment_energy(equipment); done(current)
        begin(AnalysisStage.FORECASTING_USAGE.value)
        if predictor is None: from ml.predict import EnergyPredictor; predictor = EnergyPredictor()
        forecast = predictor.predict(df)
        done(current)
        begin(AnalysisStage.CHECKING_ANOMALIES.value)
        anomalies = detect_anomalies(df.energy_kwh, forecast, df.timestamp, profile.tariff_inr_per_kwh); done(current)
        begin(AnalysisStage.FINDING_OPPORTUNITIES.value)
        opportunities = identify_energy_opportunities(anomalies, profile, equipment, profile.tariff_inr_per_kwh, emission_factor_kg_per_kwh); done(current)
        begin(AnalysisStage.OPTIMIZING_ACTIONS.value)
        optimization = optimize_equipment_schedule(profile, equipment, profile.tariff_inr_per_kwh); done(current)
        begin(AnalysisStage.CALCULATING_IMPACT.value)
        impact = calculate_impact(optimization, profile.net_built_up_area_m2, 12.0, emission_factor_kg_per_kwh); done(current)
        begin(AnalysisStage.MATCHING_INCENTIVES.value)
        incentives = find_incentives(profile.location_state, profile.building_type, equipment); done(current)
        begin(AnalysisStage.GENERATING_AI_INSIGHT.value)
        context = build_analysis_context(profile, equipment, df, forecast, anomalies, opportunities, optimization, impact, incentives)
        manager = ai_manager or AIManager(); insight = manager.interpret(context); done(current)
        begin(AnalysisStage.FINALIZING_ACTION_PLAN.value)
        plan = _action_plan(context, insight, opportunities, impact, incentives, equipment); done(current)
        return AnalysisResult(context.to_dict(), [float(x) for x in forecast], _records(anomalies),
          [o.to_dict() for o in opportunities], optimization.to_dict(), impact.to_dict(), incentives,
          insight.to_dict(), manager.last_status.to_dict(), plan, progress.to_dict())
    except Exception as exc:
        if current: progress.update(current, "failed", str(exc))
        exc.analysis_progress = progress.to_dict()
        raise
