import json

import pandas as pd
import pytest

from domain.models import BuildingProfile, Equipment
from services.ai.deterministic_provider import DeterministicProvider
from services.ai.gemini_provider import GeminiProvider
from services.ai.manager import AIManager
from services.ai.ollama_provider import OllamaProvider
from services.ai.schemas import StructuredAIInsight, StructuredAnalysisContext
from services.ai import remote_provider
from services.analysis_service import AnalysisProgress, analyze_energy, build_analysis_context
from services.incentive_service import find_incentives
from services.impact_service import calculate_impact
from services.optimization_service import optimize_equipment_schedule


def sample_context():
    return StructuredAnalysisContext({"actual_energy_kwh": 123.0}, [{"title": "Review after-hours usage", "recommended_action": "Check shutdown schedules."}], ["Meter data cannot identify equipment root cause."])


def test_environment_selects_gemini_provider(monkeypatch):
    from services.ai.gemini_provider import GeminiProvider
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    manager = AIManager()
    assert isinstance(manager.provider, GeminiProvider)


def test_environment_selects_deterministic_provider(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "deterministic")
    manager = AIManager()
    assert isinstance(manager.provider, DeterministicProvider)


def test_deterministic_provider_returns_structured_insight():
    result = DeterministicProvider().generate(sample_context())
    assert result.key_finding == "Review after-hours usage"
    assert result.recommendations == ["Check shutdown schedules."]
    assert result.confidence in {"low", "medium", "high"}


def test_ollama_unavailable_falls_back_locally(monkeypatch):
    monkeypatch.setattr(OllamaProvider, "generate", lambda *_: (_ for _ in ()).throw(ConnectionError()))
    manager = AIManager("ollama")
    result = manager.interpret(sample_context())
    assert result.key_finding == "Review after-hours usage"
    assert manager.last_status.fallback_active


def test_gemini_unavailable_falls_back_locally(monkeypatch):
    monkeypatch.setattr(GeminiProvider, "generate", lambda *_: (_ for _ in ()).throw(RuntimeError("secret should not leak")))
    manager = AIManager("gemini")
    manager.interpret(sample_context())
    assert manager.last_status.fallback_active
    assert "secret" not in manager.last_status.message


def test_malformed_provider_output_falls_back(monkeypatch):
    monkeypatch.setattr(OllamaProvider, "generate", lambda *_: StructuredAIInsight.from_dict({}))
    manager = AIManager("ollama")
    assert manager.interpret(sample_context()).key_finding == "Review after-hours usage"
    assert manager.last_status.fallback_active


def test_provider_gets_one_retry_before_fallback(monkeypatch):
    calls = []
    def malformed(self, _):
        calls.append(1)
        raise ValueError("malformed")
    monkeypatch.setattr(OllamaProvider, "generate", malformed)
    manager = AIManager("ollama")
    manager.interpret(sample_context())
    assert len(calls) == 2
    assert manager.last_status.fallback_active


def test_no_silent_provider_switching(monkeypatch):
    called = []
    monkeypatch.setattr(OllamaProvider, "generate", lambda *_: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(remote_provider.OpenAICompatibleClient, "generate", lambda *_: called.append("openai"))
    manager = AIManager("ollama")
    manager.interpret(sample_context())
    assert not called
    assert manager.last_status.provider == "ollama"


def test_ai_cannot_introduce_numerical_facts():
    class Fake:
        def generate(self, _):
            return StructuredAIInsight("Save 999 kWh", "999 kWh", [], "", "", "", "high", [])
    with pytest.raises(ValueError, match="numerical"):
        remote_provider._validate_numeric_claims(Fake().generate(None), sample_context())


def test_structured_insight_schema_validation():
    result = DeterministicProvider().generate(sample_context())
    assert StructuredAIInsight.from_dict(result.to_dict()) == result
    with pytest.raises(ValueError): StructuredAIInsight.from_dict({"summary": "bad"})


def test_progress_contract_has_supported_statuses():
    progress = AnalysisProgress()
    progress.update("VALIDATING_DATA", "running", "checking")
    assert progress.to_dict()[0]["status"] == "running"
    with pytest.raises(ValueError): progress.update("VALIDATING_DATA", "done")


def test_analysis_failure_exposes_failed_progress():
    profile = BuildingProfile("b", "Commercial Office", "Maharashtra", "Composite", 1000, 8, 18, 22, 20, 10000, 8)
    df = pd.DataFrame({"timestamp": pd.to_datetime([]), "energy_kwh": []})
    with pytest.raises(ValueError) as error:
        analyze_energy(profile, [], df)
    assert error.value.analysis_progress[0]["key"] == "VALIDATING_DATA"
    assert error.value.analysis_progress[0]["status"] == "failed"


def test_different_data_changes_analysis_context():
    profile = BuildingProfile("b", "Commercial Office", "Maharashtra", "Composite", 1000, 8, 18, 22, 20, 10000, 8)
    equipment = [Equipment("e", "Lighting", "Lights", 10, 1, 10, 22, .8, 5, 10, True)]
    opt = optimize_equipment_schedule(profile, equipment)
    impact = calculate_impact(opt, 1000)
    incentives = find_incentives(profile.location_state, profile.building_type, equipment)
    df1 = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=48, freq="h"), "energy_kwh": [20.] * 48})
    df2 = df1.copy(); df2["energy_kwh"] = [30.] * 48
    anom = pd.DataFrame({"anomaly_flag": [False] * 48})
    c1 = build_analysis_context(profile, equipment, df1, [19.] * 48, anom, [], opt, impact, incentives)
    c2 = build_analysis_context(profile, equipment, df2, [19.] * 48, anom, [], opt, impact, incentives)
    assert c1.facts["actual_energy_kwh"] != c2.facts["actual_energy_kwh"]


def test_no_opportunity_empty_state_is_explicit():
    context = StructuredAnalysisContext({"anomaly_count": 0})
    result = DeterministicProvider().generate(context)
    assert result.key_finding == "No significant efficiency opportunity was detected from the available data."


def test_end_to_end_analysis_returns_grounded_contract():
    profile = BuildingProfile("b", "Commercial Office", "Maharashtra", "Composite", 1000, 8, 18, 22, 20, 10000, 8)
    equipment = [Equipment("e", "Lighting", "Lights", 10, 1, 10, 22, .8, 5, 10, True)]
    df = pd.read_csv("data/demo/demo_office_energy.csv", nrows=96, parse_dates=["timestamp"])
    result = analyze_energy(profile, equipment, df, ai_manager=AIManager("deterministic"))
    assert result.action_plan["expected_impact"] == result.impact
    assert all(x["status"] == "complete" for x in result.progress)
    assert result.ai_provider["provider"] == "deterministic"
