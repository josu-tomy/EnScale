
"""
Unit and integration tests for Energy Opportunity identification, explainable anomaly episodes,
decision loop summaries, and optimization schedule formatting.
"""

import pytest
import pandas as pd
import numpy as np

from domain.models import BuildingProfile, Equipment, EnergyOpportunity
from services.opportunity_service import (
    explain_anomaly_episodes,
    identify_energy_opportunities,
    generate_decision_summary,
)
from services.optimization_service import optimize_equipment_schedule
from services.impact_service import calculate_impact
from services.incentive_service import find_incentives
from services.anomaly_service import detect_anomalies


@pytest.fixture
def sample_building():
    return BuildingProfile(
        building_id="BLD-TEST-OPP",
        building_type="Commercial Office",
        location_state="Maharashtra",
        climate_zone="Composite",
        net_built_up_area_m2=2500.0,
        operating_start=8,
        operating_end=18,
        operating_days_per_month=22,
        occupancy=140.0,
        monthly_budget_inr=150000.0,
        tariff_inr_per_kwh=8.50,
    )


@pytest.fixture
def sample_equipment():
    return [
        Equipment(
            equipment_id="EQ-HVAC-01",
            equipment_type="HVAC",
            equipment_name="Central Chiller",
            rated_power_kw=35.0,
            quantity=1,
            hours_per_day=9.0,
            operating_days=22,
            utilization_factor=0.80,
            minimum_hours=7.0,
            maximum_hours=9.0,
            is_flexible=True,
        ),
        Equipment(
            equipment_id="EQ-PUMP-01",
            equipment_type="Motors & Pumps",
            equipment_name="Water Circulation Pumps",
            rated_power_kw=15.0,
            quantity=2,
            hours_per_day=6.0,
            operating_days=22,
            utilization_factor=0.85,
            minimum_hours=4.0,
            maximum_hours=6.0,
            is_flexible=True,
        ),
        Equipment(
            equipment_id="EQ-LIGHT-01",
            equipment_type="Lighting",
            equipment_name="Office Lighting",
            rated_power_kw=10.0,
            quantity=1,
            hours_per_day=10.0,
            operating_days=22,
            utilization_factor=0.90,
            minimum_hours=10.0,
            maximum_hours=10.0,
            is_flexible=False,
        ),
    ]


@pytest.fixture
def sample_anomaly_df():
    # 72 hours of data with controlled anomaly from hour 19 to 23 (Friday evening after-hours)
    timestamps = pd.date_range("2026-07-03 00:00:00", periods=72, freq="h")
    actual = np.full(72, 20.0)
    pred = np.full(72, 20.0)

    # Injected Friday night spike: hours 19 to 23 (residual = 27 kW <= 35 kW chiller capacity)
    actual[19:24] = 45.0
    pred[19:24] = 18.0

    df = detect_anomalies(
        actual_energy_kwh=actual,
        predicted_energy_kwh=pred,
        timestamp=timestamps,
        tariff_inr_per_kwh=8.50,
    )
    return df


def test_explain_anomaly_episodes(sample_building, sample_anomaly_df):
    episodes = explain_anomaly_episodes(
        anomaly_df=sample_anomaly_df,
        building_profile=sample_building,
        tariff_inr_per_kwh=8.50,
        limit=3,
    )

    assert len(episodes) >= 1
    top_ep = episodes[0]
    assert "HVAC" in top_ep["category"] or "Chillers" in top_ep["category"]
    assert top_ep["duration_hours"] == 5
    assert top_ep["total_waste_kwh"] > 0
    assert top_ep["total_waste_cost_inr"] > 0
    assert "past facility closing time" in top_ep["likely_operational_condition"] or "past the facility closing time" in top_ep["likely_operational_condition"]
    assert top_ep["max_z_score"] >= 2.5


def test_identify_energy_opportunities(sample_building, sample_equipment, sample_anomaly_df):
    opps = identify_energy_opportunities(
        anomaly_df=sample_anomaly_df,
        building_profile=sample_building,
        equipment_list=sample_equipment,
        tariff_inr_per_kwh=8.50,
        emission_factor_kg_per_kwh=0.82,
    )

    assert len(opps) >= 2
    # Verify HVAC after-hours opportunity
    hvac_opp = next((o for o in opps if "HVAC" in o.category or "Cooling" in o.opportunity_name), None)
    assert hvac_opp is not None
    assert hvac_opp.estimated_avoidable_energy_kwh > 0
    assert hvac_opp.estimated_annual_cost_savings_inr > 0
    assert hvac_opp.estimated_annual_co2_impact_kg > 0
    assert len(hvac_opp.constraints) > 0
    assert "Optimization" in hvac_opp.confidence_basis or "Schedule" in hvac_opp.confidence_basis or "Model Residual" in hvac_opp.confidence_basis

    # Verify Pump shifting opportunity
    pump_opp = next((o for o in opps if "Motors & Pumps" in o.category or "Pump" in o.opportunity_name), None)
    assert pump_opp is not None
    assert ("Water Circulation Pumps" in pump_opp.opportunity_name or "Pump" in pump_opp.opportunity_name)
    assert pump_opp.estimated_annual_cost_savings_inr > 0


def test_optimization_schedule_formatting(sample_building, sample_equipment):
    res = optimize_equipment_schedule(sample_building, sample_equipment)
    assert res.is_feasible is True
    assert len(res.schedule_recommendations) == 3

    for rec in res.schedule_recommendations:
        assert "current_start_time" in rec
        assert "current_stop_time" in rec
        assert "optimized_start_time" in rec
        assert "optimized_stop_time" in rec
        assert "baseline_hours_per_day" in rec
        assert "optimized_hours_per_day" in rec
        assert rec["optimized_hours_per_day"] >= rec["minimum_hours"]


def test_decision_summary_generation(sample_building, sample_equipment, sample_anomaly_df):
    opps = identify_energy_opportunities(sample_anomaly_df, sample_building, sample_equipment)
    opt_res = optimize_equipment_schedule(sample_building, sample_equipment)
    impact = calculate_impact(opt_res, net_built_up_area_m2=sample_building.net_built_up_area_m2)
    matches = find_incentives(sample_building.location_state, sample_building.building_type, sample_equipment)

    summary = generate_decision_summary(opps, opt_res, impact, matches, sample_building)

    assert "detected_opportunity" in summary
    assert "estimated_avoidable_energy" in summary
    assert "recommended_intervention" in summary
    assert "estimated_financial_impact" in summary
    assert "potential_avoided_emissions" in summary
    assert "policy_opportunity" in summary
    assert summary["tariff_used"] == 8.50
    assert summary["co2_factor_used"] == 0.82


def test_per_equipment_equality_schedule_and_opportunities(sample_building, sample_equipment, sample_anomaly_df):
    """
    Asserts strict per-equipment equality between schedule recommendations (Improve table)
    and headline verified EnergyOpportunity cards (Action Plan screen).
    """
    tariff = 8.50
    emission_factor = 0.82
    opt_res = optimize_equipment_schedule(
        building_profile=sample_building,
        equipment_list=sample_equipment,
        tariff_inr_per_kwh=tariff,
        emission_factor_kg_per_kwh=emission_factor,
    )
    opps = identify_energy_opportunities(
        anomaly_df=sample_anomaly_df,
        building_profile=sample_building,
        equipment_list=sample_equipment,
        tariff_inr_per_kwh=tariff,
        emission_factor_kg_per_kwh=emission_factor,
    )

    flexible_recs = [r for r in opt_res.schedule_recommendations if r.get("is_flexible") and r.get("modeled_energy_savings_kwh", 0) > 0]
    verified_opps = [o for o in opps if getattr(o, "is_headline_verified", True)]

    assert len(flexible_recs) == len(verified_opps)

    for rec in flexible_recs:
        matching_opp = next((o for o in verified_opps if o.equipment_id == rec["equipment_id"]), None)
        assert matching_opp is not None, f"No opportunity card found for equipment {rec['equipment_id']}"
        assert matching_opp.annual_avoidable_energy_kwh == rec["annual_energy_savings_kwh"]
        assert matching_opp.estimated_annual_cost_savings_inr == rec["annual_cost_savings_inr"]
        assert matching_opp.estimated_annual_co2_impact_kg == rec["annual_co2_savings_kg"]
        assert matching_opp.estimated_avoidable_energy_kwh == rec["modeled_energy_savings_kwh"]


def test_no_unsupported_causal_wording_in_opportunities(sample_building, sample_equipment, sample_anomaly_df):
    """
    Asserts that banned unsupported causal phrasing is absent from all opportunities,
    and that the honest standardized wording is strictly used.
    """
    opps = identify_energy_opportunities(sample_anomaly_df, sample_building, sample_equipment)
    banned_phrases = [
        "pre-cooling demand",
        "unnecessary runtime",
        "hydraulic demand satisfied",
        "avoid surge episodes",
        "unneeded idling",
    ]

    for opp in opps:
        for text in [opp.current_condition, opp.expected_condition, opp.possible_intervention]:
            for banned in banned_phrases:
                assert banned not in text.lower(), f"Banned phrase '{banned}' found in opportunity {opp.opportunity_id}: {text}"

    verified_opps = [o for o in opps if getattr(o, "is_headline_verified", True)]
    for opp in verified_opps:
        assert "runtime reduced to the minimum you entered; process and comfort impact not modeled — verify on site." in opp.expected_condition


def test_anomaly_attribution_rated_capacity_check(sample_building):
    """
    Asserts anomaly hypotheses do not cite a load magnitude as matching rated capacity
    unless the residual kW is <= the listed equipment's rated kW; otherwise mark
    the category 'not attributable from meter data'.
    """
    equipment = [
        Equipment(
            equipment_id="EQ-COMP-01",
            equipment_type="Air Compressors",
            equipment_name="Air Compressors",
            rated_power_kw=18.0,
            quantity=1,
            hours_per_day=8.0,
            operating_days=22,
            utilization_factor=0.80,
            minimum_hours=5.0,
            maximum_hours=8.0,
            is_flexible=True,
        ),
        Equipment(
            equipment_id="EQ-HVAC-01",
            equipment_type="HVAC",
            equipment_name="Central Chiller",
            rated_power_kw=35.0,
            quantity=1,
            hours_per_day=9.0,
            operating_days=22,
            utilization_factor=0.80,
            minimum_hours=7.0,
            maximum_hours=9.0,
            is_flexible=True,
        ),
    ]

    # Create daytime episode during operating hours with avg_residual = 34.2 kW
    # 34.2 kW exceeds compressor rated capacity (18.0 kW) -> Must be "not attributable from meter data"
    timestamps = pd.date_range("2026-08-24 08:00:00", periods=10, freq="h")
    actual = np.full(10, 80.0)
    actual[4:10] = 191.1  # 12:00 - 17:00
    pred = np.full(10, 80.0)
    pred[4:10] = 156.9

    anom_df = detect_anomalies(actual, pred, timestamps, tariff_inr_per_kwh=8.50)
    episodes = explain_anomaly_episodes(anom_df, sample_building, limit=1, equipment_list=equipment)

    assert len(episodes) == 1
    assert episodes[0]["category"] == "not attributable from meter data"
    assert "exceeds listed equipment rated capacity" in episodes[0]["likely_operational_condition"]

    # Now create after-hours episode with avg_residual = 16.2 kW <= 35.0 kW chiller rated capacity
    df_hvac = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-03 19:00:00", periods=5, freq="h"),
        "actual_energy_kwh": [86.0] * 5,
        "predicted_energy_kwh": [69.8] * 5,
        "forecast_error_kwh": [16.2] * 5,
        "waste_kwh": [16.2] * 5,
        "potential_waste_cost_inr": [16.2 * 8.5] * 5,
        "robust_z_score": [3.5] * 5,
        "anomaly_flag": [True] * 5,
    })
    episodes_hvac = explain_anomaly_episodes(df_hvac, sample_building, limit=1, equipment_list=equipment)

    assert len(episodes_hvac) == 1
    assert episodes_hvac[0]["category"] == "HVAC / Chillers"
    assert "35.0 kW rated capacity" in episodes_hvac[0]["confidence"]


def test_unverified_status_in_reference_files():
    """
    Asserts no entry in incentives.json or emission_factors.json is marked verified
    unless verified_by_human: true is explicitly present.
    """
    import json
    from pathlib import Path

    inc_path = Path("data/reference/incentives.json")
    if inc_path.exists():
        with open(inc_path, "r", encoding="utf-8") as f:
            incentives = json.load(f)
        for inc in incentives:
            assert inc.get("verified_by_human") is False
            assert "UNVERIFIED" in inc.get("verification_status", "")
            assert inc.get("last_verified") is None

    ef_path = Path("data/reference/emission_factors.json")
    if ef_path.exists():
        with open(ef_path, "r", encoding="utf-8") as f:
            ef_data = json.load(f)
        for region, ef in ef_data.get("factors", {}).items():
            assert ef.get("verified_by_human") is False
            assert "UNVERIFIED" in ef.get("verification_status", "")


def test_proportional_co2_recalculation(sample_building, sample_equipment):
    """
    Asserts changing the emission factor changes the CO2 total proportionally.
    """
    opt_res = optimize_equipment_schedule(sample_building, sample_equipment)

    impact_082 = calculate_impact(opt_res, net_built_up_area_m2=sample_building.net_built_up_area_m2, emission_factor_kg_per_kwh=0.82)
    impact_050 = calculate_impact(opt_res, net_built_up_area_m2=sample_building.net_built_up_area_m2, emission_factor_kg_per_kwh=0.50)

    ratio = impact_050.co2_avoided_kg / impact_082.co2_avoided_kg
    expected_ratio = 0.50 / 0.82
    assert abs(ratio - expected_ratio) < 1e-4


def test_needed_until_closing_and_opening_constraints(sample_building):
    """
    Asserts that equipment marked needed_until_closing cannot stop before operating_end,
    and equipment marked needed_from_opening starts at operating_start.
    """
    eq = [
        Equipment(
            equipment_id="EQ-TEST-01",
            equipment_type="HVAC",
            equipment_name="Closing Required Unit",
            rated_power_kw=20.0,
            quantity=1,
            hours_per_day=10.0,
            operating_days=22,
            utilization_factor=0.8,
            minimum_hours=6.0,
            maximum_hours=10.0,
            is_flexible=True,
            needed_until_closing=True,
            needed_from_opening=False,
        ),
        Equipment(
            equipment_id="EQ-TEST-02",
            equipment_type="HVAC",
            equipment_name="Opening Required Unit",
            rated_power_kw=20.0,
            quantity=1,
            hours_per_day=10.0,
            operating_days=22,
            utilization_factor=0.8,
            minimum_hours=6.0,
            maximum_hours=10.0,
            is_flexible=True,
            needed_until_closing=False,
            needed_from_opening=True,
        ),
    ]

    opt_res = optimize_equipment_schedule(sample_building, eq)
    recs = opt_res.schedule_recommendations

    rec_closing = next(r for r in recs if r["equipment_id"] == "EQ-TEST-01")
    rec_opening = next(r for r in recs if r["equipment_id"] == "EQ-TEST-02")

    assert rec_closing["optimized_stop_time"] == f"{sample_building.operating_end:02d}:00"
    assert rec_opening["optimized_start_time"] == f"{sample_building.operating_start:02d}:00"


def test_reconciled_headline_savings_equality(sample_building, sample_equipment, sample_anomaly_df):
    """
    Asserts strict numerical equality between the sum of verified opportunities
    and the impact_service annual totals.
    """
    opt_res = optimize_equipment_schedule(sample_building, sample_equipment)
    impact = calculate_impact(opt_res, net_built_up_area_m2=sample_building.net_built_up_area_m2)
    opps = identify_energy_opportunities(sample_anomaly_df, sample_building, sample_equipment)

    verified_opps = [o for o in opps if getattr(o, "is_headline_verified", True)]
    sum_opp_kwh = sum(o.annual_avoidable_energy_kwh for o in verified_opps)
    sum_opp_cost = sum(o.estimated_annual_cost_savings_inr for o in verified_opps)
    sum_opp_co2 = sum(o.estimated_annual_co2_impact_kg for o in verified_opps)

    assert round(sum_opp_kwh, 1) == round(impact.annual_energy_savings, 1)
    assert round(sum_opp_cost, 0) == round(impact.annual_cost_savings, 0)
    assert round(sum_opp_co2, 1) == round(impact.co2_avoided_kg, 1)


def test_injected_anomaly_recall_benchmark():
    """
    Asserts calculate_injected_anomaly_recall correctly calculates recall for demo data.
    """
    from services.ingestion_service import load_energy_csv
    from ml.predict import EnergyPredictor
    from services.anomaly_service import calculate_injected_anomaly_recall

    df = load_energy_csv("data/demo/demo_office_energy.csv")
    preds = EnergyPredictor().predict(df)
    anom_df = detect_anomalies(df["energy_kwh"], preds, df["timestamp"], tariff_inr_per_kwh=8.50)

    recall_info = calculate_injected_anomaly_recall(anom_df)
    assert recall_info["total_injected_episodes"] == 5
    assert recall_info["detected_injected_episodes"] == 5
    assert recall_info["episode_recall_pct"] == 100.0


def test_setup_data_manifest_and_generation():
    """
    Asserts demo dataset manifest does not claim real-world measurements or uncalibrated benchmarks.
    """
    import json
    from pathlib import Path

    manifest_path = Path("data/demo/demo_manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["is_real_world_measurement"] is False
        assert manifest["rows_count"] == 1488
        assert "simulated" in manifest["dataset_label"].lower()


def test_binding_constraint_populated_in_opportunities(sample_building, sample_equipment, sample_anomaly_df):
    """
    Asserts every verified opportunity has a descriptive binding constraint.
    """
    opps = identify_energy_opportunities(sample_anomaly_df, sample_building, sample_equipment)
    verified_opps = [o for o in opps if getattr(o, "is_headline_verified", True)]

    for opp in verified_opps:
        assert len(opp.binding_constraint) > 0
        assert "minimum runtime" in opp.binding_constraint.lower()


def test_equipment_coverage_calculation(sample_building, sample_equipment):
    """
    Asserts equipment coverage calculation logic executes accurately.
    """
    from services.ingestion_service import load_energy_csv
    from services.baseline_service import calculate_equipment_energy, calculate_equipment_coverage

    df = load_energy_csv("data/demo/demo_office_energy.csv")
    eq_calc = calculate_equipment_energy(sample_equipment)
    cov = calculate_equipment_coverage(eq_calc["total_energy"], df)

    assert "coverage_pct" in cov
    assert "unmodeled_pct" in cov
    assert cov["coverage_pct"] + cov["unmodeled_pct"] == pytest.approx(100.0, abs=0.1)


