"""
Comprehensive test suite for constrained energy optimization and the Optimization Test Matrix.

Covers:
TEST 1: No flexible equipment (optimized == baseline)
TEST 2: Flexible equipment (valid optimized schedule)
TEST 3: Impossible constraints (no feasible solution)
TEST 4: All output schedules (zero constraint violations)
TEST 5: Cost calculation (cost = kWh * tariff within tolerance)
"""

import pytest
from domain.models import Equipment, BuildingProfile, OptimizationResult
from services.optimization_service import (
    compute_modeled_savings,
    optimize_equipment_schedule,
    InfeasibleConstraintError,
    OptimizationService,
)


@pytest.fixture
def standard_building() -> BuildingProfile:
    return BuildingProfile(
        building_id="BLD-TEST-01",
        building_type="Commercial Office",
        location_state="Maharashtra",
        climate_zone="Composite",
        net_built_up_area_m2=2500.0,
        operating_start=8,
        operating_end=18,  # 10-hour operating window
        operating_days_per_month=22,
        occupancy=120.0,
        monthly_budget_inr=150000.0,
        tariff_inr_per_kwh=8.50,
    )


def test_savings_contract_formula():
    """Verifies canonical modeled savings formulas."""
    res = compute_modeled_savings(
        baseline_energy_kwh=1000.0,
        optimized_energy_kwh=850.0,
        tariff_inr_per_kwh=8.0,
    )
    assert isinstance(res, OptimizationResult)
    assert res.energy_savings_kwh == 150.0
    assert res.baseline_cost_inr == 8000.0
    assert res.optimized_cost_inr == 6800.0
    assert res.cost_savings_inr == 1200.0
    assert res.savings_pct == 15.0
    assert "Projected savings under the modeled operating constraints" in res.savings_statement


def test_optimization_matrix_test1_no_flexible_equipment(standard_building):
    """
    TEST 1: No flexible equipment.
    Expected: optimized == baseline, energy_savings_kwh == 0.0, cost_savings_inr == 0.0.
    """
    eq1 = Equipment(
        equipment_id="EQ-FIX-01",
        equipment_type="Lighting",
        equipment_name="Office Lighting",
        rated_power_kw=10.0,
        quantity=2,
        hours_per_day=9.0,
        operating_days=22,
        utilization_factor=0.9,
        minimum_hours=8.0,
        maximum_hours=10.0,
        is_flexible=False,
    )
    eq2 = Equipment(
        equipment_id="EQ-FIX-02",
        equipment_type="IT Equipment",
        equipment_name="Server Room IT",
        rated_power_kw=5.0,
        quantity=1,
        hours_per_day=10.0,
        operating_days=22,
        utilization_factor=1.0,
        minimum_hours=10.0,
        maximum_hours=10.0,
        is_flexible=False,
    )

    res = optimize_equipment_schedule(standard_building, [eq1, eq2])

    assert res.is_feasible is True
    assert res.optimized_energy_kwh == res.baseline_energy_kwh
    assert res.optimized_cost_inr == res.baseline_cost_inr
    assert res.energy_savings_kwh == 0.0
    assert res.cost_savings_inr == 0.0
    assert res.savings_pct == 0.0
    assert "No modeled savings" in res.status_message


def test_optimization_matrix_test2_flexible_equipment(standard_building):
    """
    TEST 2: Flexible equipment.
    Expected: valid optimized schedule, modeled savings > 0.
    """
    eq_flex = Equipment(
        equipment_id="EQ-FLEX-01",
        equipment_type="HVAC",
        equipment_name="Modulating Chiller",
        rated_power_kw=30.0,
        quantity=1,
        hours_per_day=10.0,
        operating_days=22,
        utilization_factor=0.8,
        minimum_hours=6.0,
        maximum_hours=10.0,
        is_flexible=True,
    )

    res = optimize_equipment_schedule(standard_building, [eq_flex])

    assert res.is_feasible is True
    assert res.optimized_energy_kwh < res.baseline_energy_kwh
    assert res.energy_savings_kwh > 0.0
    assert res.cost_savings_inr > 0.0
    assert len(res.schedule_recommendations) == 1
    rec = res.schedule_recommendations[0]
    assert rec["is_flexible"] is True
    assert rec["optimized_hours_per_day"] == 6.0
    assert rec["modeled_energy_savings_kwh"] > 0.0


def test_optimization_matrix_test3_impossible_constraints(standard_building):
    """
    TEST 3: Impossible constraints.
    Expected: no feasible solution (InfeasibleConstraintError raised).
    """
    # Case A: minimum_hours > maximum_hours
    eq_impossible_bounds = Equipment(
        equipment_id="EQ-IMP-01",
        equipment_type="HVAC",
        equipment_name="Broken Constraint Equipment",
        rated_power_kw=20.0,
        quantity=1,
        hours_per_day=8.0,
        operating_days=22,
        utilization_factor=1.0,
        minimum_hours=12.0,
        maximum_hours=8.0,  # Contradiction: min > max
        is_flexible=True,
    )
    with pytest.raises(InfeasibleConstraintError, match="No feasible solution"):
        optimize_equipment_schedule(standard_building, [eq_impossible_bounds])

    # Case B: minimum runtime exceeds building operating window (window is 10 hours: 8 to 18)
    eq_exceeds_window = Equipment(
        equipment_id="EQ-IMP-02",
        equipment_type="HVAC",
        equipment_name="Over-window Equipment",
        rated_power_kw=20.0,
        quantity=1,
        hours_per_day=10.0,
        operating_days=22,
        utilization_factor=1.0,
        minimum_hours=14.0,  # 14 hours > 10 hour window
        maximum_hours=16.0,
        is_flexible=True,
    )
    with pytest.raises(InfeasibleConstraintError, match="exceeds building operating window"):
        optimize_equipment_schedule(standard_building, [eq_exceeds_window])


def test_optimization_matrix_test4_zero_constraint_violations(standard_building):
    """
    TEST 4: All output schedules.
    Expected: zero constraint violations across all flexible and non-flexible assets.
    """
    building_window = float(standard_building.operating_end - standard_building.operating_start)  # 10.0h

    equipment_suite = [
        Equipment(
            equipment_id="EQ-01",
            equipment_type="HVAC",
            equipment_name="Main Chiller",
            rated_power_kw=45.0,
            quantity=2,
            hours_per_day=9.5,
            operating_days=22,
            utilization_factor=0.75,
            minimum_hours=6.5,
            maximum_hours=10.0,
            is_flexible=True,
        ),
        Equipment(
            equipment_id="EQ-02",
            equipment_type="Motors & Pumps",
            equipment_name="Water Pump",
            rated_power_kw=15.0,
            quantity=2,
            hours_per_day=8.0,
            operating_days=22,
            utilization_factor=0.85,
            minimum_hours=5.0,
            maximum_hours=9.0,
            is_flexible=True,
        ),
        Equipment(
            equipment_id="EQ-03",
            equipment_type="Lighting",
            equipment_name="Core Lights",
            rated_power_kw=8.0,
            quantity=1,
            hours_per_day=10.0,
            operating_days=22,
            utilization_factor=0.95,
            minimum_hours=10.0,
            maximum_hours=10.0,
            is_flexible=False,
        ),
    ]

    res = optimize_equipment_schedule(standard_building, equipment_suite)

    assert res.is_feasible is True
    assert len(res.schedule_recommendations) == 3

    for eq, rec in zip(equipment_suite, res.schedule_recommendations):
        opt_h = rec["optimized_hours_per_day"]
        # Rule 1: Must satisfy equipment minimum hours
        assert opt_h >= eq.minimum_hours, f"Violated minimum_hours for {eq.equipment_id}"
        # Rule 2: Must satisfy equipment maximum hours
        assert opt_h <= eq.maximum_hours, f"Violated maximum_hours for {eq.equipment_id}"
        # Rule 3: Must not exceed building operating window
        assert opt_h <= building_window, f"Exceeded building window for {eq.equipment_id}"
        # Rule 4: Non-flexible items must be unchanged
        if not eq.is_flexible:
            assert opt_h == eq.hours_per_day, f"Non-flexible item altered for {eq.equipment_id}"


def test_optimization_matrix_test5_cost_calculation(standard_building):
    """
    TEST 5: Cost calculation.
    Expected: cost = kWh * tariff within numerical tolerance (1e-2).
    """
    tariff = standard_building.tariff_inr_per_kwh
    eq = Equipment(
        equipment_id="EQ-COST-01",
        equipment_type="HVAC",
        equipment_name="Chiller Test",
        rated_power_kw=25.0,
        quantity=1,
        hours_per_day=10.0,
        operating_days=20,
        utilization_factor=0.8,
        minimum_hours=7.0,
        maximum_hours=10.0,
        is_flexible=True,
    )

    res = optimize_equipment_schedule(standard_building, [eq], tariff_inr_per_kwh=tariff)

    # Base cost check: kWh * tariff
    expected_base_cost = res.baseline_energy_kwh * tariff
    assert abs(res.baseline_cost_inr - expected_base_cost) < 1e-2

    # Optimized cost check: kWh * tariff
    expected_opt_cost = res.optimized_energy_kwh * tariff
    assert abs(res.optimized_cost_inr - expected_opt_cost) < 1e-2

    # Cost savings check: energy_savings_kwh * tariff
    expected_cost_savings = res.energy_savings_kwh * tariff
    assert abs(res.cost_savings_inr - expected_cost_savings) < 1e-2
