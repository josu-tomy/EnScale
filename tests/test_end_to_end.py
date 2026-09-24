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
from io import StringIO

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
from services.impact_service import ImpactService, calculate_impact
from services.optimization_service import compute_modeled_savings, optimize_equipment_schedule
from services.ingestion_service import load_demo_energy_data, load_energy_csv
from services.baseline_service import calculate_equipment_energy
from ml.predict import EnergyPredictor


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


def test_execution_chain_1_demo_pipeline():
    """
    Execution Chain 1:
    Demo data -> forecast -> anomaly -> optimization -> savings -> CO2 -> EPI/MEPI.
    """
    from services.anomaly_service import detect_anomalies

    # 1. Demo data
    demo_df = load_demo_energy_data()
    assert not demo_df.empty
    assert "timestamp" in demo_df.columns
    assert "energy_kwh" in demo_df.columns

    # 2. ML Forecast
    predictor = EnergyPredictor()
    assert predictor.is_loaded()
    preds = predictor.predict(demo_df)
    assert len(preds) == len(demo_df)

    # 3. Anomaly detection
    anomaly_df = detect_anomalies(
        actual_energy_kwh=demo_df["energy_kwh"],
        predicted_energy_kwh=preds,
        timestamp=demo_df["timestamp"],
    )
    assert len(anomaly_df) == len(demo_df)
    assert "anomaly_flag" in anomaly_df.columns
    assert "forecast_error_kwh" in anomaly_df.columns
    assert "forecast_error_pct" in anomaly_df.columns
    assert "severity" in anomaly_df.columns
    anomalies_detected = anomaly_df[anomaly_df["anomaly_flag"] == True]
    assert len(anomalies_detected) > 0

    # 4. Constrained Optimization
    b_profile = BuildingProfile(
        building_id="BLD-DEMO-01",
        building_type="Commercial Office",
        location_state="Karnataka",
        climate_zone="Composite",
        net_built_up_area_m2=2500.0,
        operating_start=8,
        operating_end=18,
        operating_days_per_month=22,
        occupancy=150.0,
        monthly_budget_inr=200000.0,
        tariff_inr_per_kwh=8.0,
    )
    eq_list = [
        Equipment(
            equipment_id="EQ-PUMP-01",
            equipment_type="pump",
            equipment_name="Water Transfer Pump",
            rated_power_kw=15.0,
            quantity=2,
            hours_per_day=6.0,
            operating_days=22,
            utilization_factor=0.85,
            minimum_hours=3.0,
            maximum_hours=6.0,
            is_flexible=True,
        ),
        Equipment(
            equipment_id="EQ-LIGHT-01",
            equipment_type="lighting",
            equipment_name="Office Lighting",
            rated_power_kw=10.0,
            quantity=1,
            hours_per_day=10.0,
            operating_days=22,
            utilization_factor=0.9,
            minimum_hours=10.0,
            maximum_hours=10.0,
            is_flexible=False,
        ),
    ]
    opt_result = optimize_equipment_schedule(
        building_profile=b_profile,
        equipment_list=eq_list,
        tariff_inr_per_kwh=8.0,
    )
    assert opt_result.is_feasible is True
    assert opt_result.baseline_energy_kwh > 0
    assert opt_result.optimized_energy_kwh <= opt_result.baseline_energy_kwh
    assert opt_result.energy_savings_kwh >= 0
    assert opt_result.cost_savings_inr >= 0
    assert "Guaranteed savings" not in opt_result.savings_statement
    assert "Actual savings" not in opt_result.savings_statement

    # 5. CO2 & Impact & EPI/MEPI
    impact = calculate_impact(
        optimization_result=opt_result,
        net_built_up_area_m2=b_profile.net_built_up_area_m2,
        months_per_year=12.0,
    )
    assert impact.monthly_energy_savings == opt_result.energy_savings_kwh
    assert impact.annual_energy_savings == opt_result.energy_savings_kwh * 12.0
    assert impact.co2_factor_used == 0.82
    assert "Central Electricity Authority" in impact.co2_factor_statement
    assert impact.mepi_kwh_m2_year > 0
    assert impact.measured_epi_status == "Insufficient data to calculate measured EPI."


def test_execution_chain_2_user_data_pipeline():
    """
    Execution Chain 2:
    User data -> forecast -> anomaly.
    """
    from services.anomaly_service import detect_anomalies

    csv_content = (
        "timestamp,energy_kwh,temperature_c,occupancy,equipment_load_kw\n"
        "2026-03-01 08:00:00,45.2,24.5,50,40.0\n"
        "2026-03-01 09:00:00,68.5,26.0,120,60.0\n"
        "2026-03-01 10:00:00,72.1,28.2,150,65.0\n"
        "2026-03-01 11:00:00,74.0,29.1,150,66.0\n"
        "2026-03-01 12:00:00,71.8,30.0,140,64.0\n"
        "2026-03-01 13:00:00,62.0,30.5,100,55.0\n"
        "2026-03-01 14:00:00,70.5,30.8,145,63.0\n"
        "2026-03-01 15:00:00,73.0,30.2,150,65.0\n"
        "2026-03-01 16:00:00,69.4,29.5,140,62.0\n"
        "2026-03-01 17:00:00,55.2,28.0,80,48.0\n"
        "2026-03-01 18:00:00,32.0,26.5,20,25.0\n"
        "2026-03-01 19:00:00,18.5,25.0,5,15.0\n"
    )
    user_df = load_energy_csv(StringIO(csv_content))
    assert not user_df.empty
    assert len(user_df) == 12

    # 2. ML Forecast
    predictor = EnergyPredictor()
    preds = predictor.predict(user_df)
    assert len(preds) == 12

    # 3. Anomaly detection
    anomaly_df = detect_anomalies(
        actual_energy_kwh=user_df["energy_kwh"],
        predicted_energy_kwh=preds,
        timestamp=user_df["timestamp"],
    )
    assert len(anomaly_df) == 12
    assert "forecast_error_kwh" in anomaly_df.columns
    assert "forecast_error_pct" in anomaly_df.columns
    assert "anomaly_flag" in anomaly_df.columns
    assert "severity" in anomaly_df.columns


def test_execution_chain_3_equipment_pipeline():
    """
    Execution Chain 3:
    Equipment mode -> baseline -> optimization -> savings.
    """
    # 1. Equipment list
    eq_list = [
        Equipment(
            equipment_id="EQ-CHILLER-01",
            equipment_type="hvac",
            equipment_name="Water Chiller",
            rated_power_kw=45.0,
            quantity=1,
            hours_per_day=10.0,
            operating_days=26,
            utilization_factor=0.8,
            minimum_hours=8.0,
            maximum_hours=10.0,
            is_flexible=True,
        ),
        Equipment(
            equipment_id="EQ-EXHAUST-01",
            equipment_type="ventilation",
            equipment_name="Exhaust Fans",
            rated_power_kw=5.5,
            quantity=3,
            hours_per_day=12.0,
            operating_days=26,
            utilization_factor=0.9,
            minimum_hours=12.0,
            maximum_hours=12.0,
            is_flexible=False,
        ),
    ]

    # 2. Equipment Baseline
    baseline_summary = calculate_equipment_energy(eq_list)
    assert baseline_summary["total_energy_kwh"] > 0
    assert len(baseline_summary["equipment_details"]) == 2
    for item in baseline_summary["equipment_details"]:
        assert "calculation_trace" in item
        assert "estimated_energy_kwh" in item

    # 3. Constrained Optimization
    b_profile = BuildingProfile(
        building_id="BLD-IND-01",
        building_type="Small Industrial",
        location_state="Gujarat",
        climate_zone="Hot and Dry",
        net_built_up_area_m2=3500.0,
        operating_start=7,
        operating_end=19,
        operating_days_per_month=26,
        occupancy=50.0,
        monthly_budget_inr=300000.0,
        tariff_inr_per_kwh=9.2,
    )
    opt_result = optimize_equipment_schedule(
        building_profile=b_profile,
        equipment_list=eq_list,
        tariff_inr_per_kwh=9.2,
    )

    # 4. Savings contract & non-flexible preservation
    assert opt_result.is_feasible is True
    assert opt_result.baseline_energy_kwh == baseline_summary["total_energy_kwh"]
    assert opt_result.energy_savings_kwh == round(opt_result.baseline_energy_kwh - opt_result.optimized_energy_kwh, 2)
    assert opt_result.cost_savings_inr == round(opt_result.baseline_cost_inr - opt_result.optimized_cost_inr, 2)
    assert "Guaranteed" not in opt_result.savings_statement
    assert "Actual savings" not in opt_result.savings_statement

