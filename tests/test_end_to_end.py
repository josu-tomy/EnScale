"""
End-to-end integration and foundation verification tests for EnScale.

Covers:
- Package & module imports
- Domain model construction
- Building validation
- Equipment validation
- Energy dataframe validation
- Pipeline integration pass-through
"""

import pandas as pd
import pytest

from domain.models import (
    BuildingProfile,
    Equipment,
    EnergyRecord,
    ForecastResult,
    AnomalyResult,
    OptimizationResult,
    IncentiveMatch,
    ImpactResult,
)
from services.validation_service import (
    validate_building_profile,
    validate_equipment,
    validate_energy_dataframe,
    validate_incentive_record,
)
from services.impact_service import ImpactService
from services.optimization_service import compute_modeled_savings


def test_imports():
    """Verifies that all project modules import cleanly without circular or broken dependencies."""
    import config.constants
    import config.settings
    import domain.enums
    import domain.models
    import domain.schemas
    import services.validation_service
    import services.ingestion_service
    import services.baseline_service
    import services.forecast_service
    import services.anomaly_service
    import services.optimization_service
    import services.impact_service
    import services.incentive_service
    import ml.preprocessing
    import ml.train
    import ml.predict
    import ml.evaluation
    import app

    assert config.constants.APP_NAME == "EnScale"


def test_domain_model_construction():
    """Verifies instantiation and serialization of all 8 domain models."""
    building = BuildingProfile(
        building_id="BLD-01",
        building_type="Commercial Office",
        location_state="Karnataka",
        climate_zone="Moderate",
        net_built_up_area_m2=5000.0,
        operating_start=8,
        operating_end=20,
        operating_days_per_month=24,
        occupancy=250.0,
        monthly_budget_inr=300000.0,
        tariff_inr_per_kwh=8.0,
    )
    assert building.net_built_up_area_m2 == 5000.0

    eq = Equipment(
        equipment_id="EQ-01",
        equipment_type="HVAC",
        equipment_name="Chiller Plant",
        rated_power_kw=100.0,
        quantity=2,
        hours_per_day=12.0,
        operating_days=24,
        utilization_factor=0.75,
        minimum_hours=8.0,
        maximum_hours=14.0,
        is_flexible=True,
    )
    assert eq.rated_power_kw == 100.0

    rec = EnergyRecord(
        timestamp="2026-08-01 12:00:00",
        energy_kwh=145.2,
        temperature_c=31.0,
        occupancy=200.0,
        equipment_load_kw=140.0,
    )
    assert rec.energy_kwh == 145.2

    forecast = ForecastResult(
        timestamp="2026-08-01 12:00:00",
        predicted_energy_kwh=140.0,
        actual_energy_kwh=145.2,
        forecast_error_kwh=5.2,
        forecast_error_pct=3.58,
    )
    assert forecast.predicted_energy_kwh == 140.0

    anomaly = AnomalyResult(
        timestamp="2026-08-01 12:00:00",
        actual_energy_kwh=145.2,
        predicted_energy_kwh=140.0,
        is_anomaly=False,
    )
    assert anomaly.is_anomaly is False

    opt = OptimizationResult(
        baseline_energy_kwh=1000.0,
        optimized_energy_kwh=850.0,
        energy_savings_kwh=150.0,
        baseline_cost_inr=8000.0,
        optimized_cost_inr=6800.0,
        cost_savings_inr=1200.0,
        savings_pct=15.0,
    )
    assert opt.energy_savings_kwh == 150.0

    inc = IncentiveMatch(
        incentive_id="INC-01",
        program_name="State DSM",
        target_equipment_type="HVAC",
        eligibility_status="eligible",
        estimated_rebate_inr=50000.0,
    )
    assert inc.estimated_rebate_inr == 50000.0

    impact = ImpactResult(
        baseline_energy_kwh=1000.0,
        optimized_energy_kwh=850.0,
        energy_savings_kwh=150.0,
        cost_savings_inr=1200.0,
        co2_reduction_kg=123.0,
        annual_energy_kwh=10200.0,
        epi_kwh_m2_year=2.04,
    )
    assert impact.epi_kwh_m2_year == 2.04


def test_building_validation():
    valid_building = BuildingProfile(
        building_id="BLD-02",
        building_type="Retail / Mall",
        location_state="Delhi",
        climate_zone="Composite",
        net_built_up_area_m2=12000.0,
        operating_start=10,
        operating_end=22,
        operating_days_per_month=30,
        occupancy=1000.0,
        monthly_budget_inr=800000.0,
        tariff_inr_per_kwh=10.5,
    )
    val = validate_building_profile(valid_building)
    assert val.is_valid is True

    alias_dict = valid_building.to_dict()
    del alias_dict["net_built_up_area_m2"]
    alias_dict["area"] = 12000.0
    val_alias = validate_building_profile(alias_dict)
    assert val_alias.is_valid is False
    assert any("Prohibited alias 'area'" in err for err in val_alias.errors)

    neg_dict = valid_building.to_dict()
    neg_dict["net_built_up_area_m2"] = -10.0
    val_neg = validate_building_profile(neg_dict)
    assert val_neg.is_valid is False

    invalid_hours = valid_building.to_dict()
    invalid_hours["operating_start"] = 20
    invalid_hours["operating_end"] = 10
    val_hours = validate_building_profile(invalid_hours)
    assert val_hours.is_valid is False


def test_equipment_validation():
    valid_eq = Equipment(
        equipment_id="EQ-PUMP-01",
        equipment_type="Motors & Pumps",
        equipment_name="Water Pump",
        rated_power_kw=15.0,
        quantity=3,
        hours_per_day=8.0,
        operating_days=25,
        utilization_factor=0.85,
        minimum_hours=4.0,
        maximum_hours=10.0,
        is_flexible=True,
    )
    val = validate_equipment(valid_eq)
    assert val.is_valid is True

    alias_eq = valid_eq.to_dict()
    del alias_eq["rated_power_kw"]
    alias_eq["power"] = 15.0
    val_alias = validate_equipment(alias_eq)
    assert val_alias.is_valid is False
    assert any("Prohibited alias 'power'" in err for err in val_alias.errors)

    invalid_util = valid_eq.to_dict()
    invalid_util["utilization_factor"] = 1.5
    val_util = validate_equipment(invalid_util)
    assert val_util.is_valid is False


def test_energy_dataframe_validation():
    df = pd.DataFrame({
        "timestamp": ["2026-08-01 00:00:00", "2026-08-01 01:00:00"],
        "energy_kwh": [50.5, 48.2],
        "temperature_c": [28.0, 27.5],
    })
    val = validate_energy_dataframe(df)
    assert val.is_valid is True

    df_alias = pd.DataFrame({
        "timestamp": ["2026-08-01 00:00:00"],
        "energy": [50.5],
    })
    val_alias = validate_energy_dataframe(df_alias)
    assert val_alias.is_valid is False
    assert any("Prohibited alias column 'energy'" in err for err in val_alias.errors)

    df_no_ts = pd.DataFrame({"energy_kwh": [50.5]})
    val_no_ts = validate_energy_dataframe(df_no_ts)
    assert val_no_ts.is_valid is False

    df_neg = pd.DataFrame({
        "timestamp": ["2026-08-01 00:00:00"],
        "energy_kwh": [-10.0],
    })
    val_neg = validate_energy_dataframe(df_neg)
    assert val_neg.is_valid is False
    assert any("negative" in err for err in val_neg.errors)


def test_pipeline_pass_through():
    opt_result = compute_modeled_savings(
        baseline_energy_kwh=10000.0,
        optimized_energy_kwh=8500.0,
        tariff_inr_per_kwh=8.5,
    )
    impact_service = ImpactService(emission_factor_kg_per_kwh=0.82)
    impact = impact_service.compute_impact(
        optimization_result=opt_result,
        net_built_up_area_m2=2000.0,
        annualization_factor=12.0,
    )

    assert impact.energy_savings_kwh == 1500.0
    assert impact.cost_savings_inr == 12750.0
    assert impact.co2_reduction_kg == 1500.0 * 0.82
    assert impact.annual_energy_kwh == 8500.0 * 12.0
    assert impact.epi_kwh_m2_year == (8500.0 * 12.0) / 2000.0
