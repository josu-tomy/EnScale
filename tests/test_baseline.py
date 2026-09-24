"""
Tests for baseline calculations, canonical equipment energy formula, and EPI normalization.
"""

import pytest
import pandas as pd

from domain.models import Equipment
from services.baseline_service import (
    calculate_equipment_energy_kwh,
    calculate_epi,
    BaselineService,
)


def test_equipment_energy_formula():
    """
    Verifies canonical formula:
    estimated_energy_kwh = rated_power_kw * quantity * hours_per_day * operating_days * utilization_factor
    """
    rated_power_kw = 50.0
    quantity = 2
    hours_per_day = 10.0
    operating_days = 25
    utilization_factor = 0.80

    expected = 50.0 * 2 * 10.0 * 25 * 0.80  # 20,000 kWh
    actual = calculate_equipment_energy_kwh(
        rated_power_kw=rated_power_kw,
        quantity=quantity,
        hours_per_day=hours_per_day,
        operating_days=operating_days,
        utilization_factor=utilization_factor,
    )
    assert actual == expected
    assert actual == 20000.0


def test_epi_calculation():
    """
    Verifies canonical EPI calculation:
    EPI = annual_energy_kwh / net_built_up_area_m2 (kWh/m2/year)
    """
    annual_energy_kwh = 300000.0
    net_built_up_area_m2 = 2000.0

    expected_epi = 150.0
    actual_epi = calculate_epi(annual_energy_kwh, net_built_up_area_m2)
    assert actual_epi == expected_epi

    with pytest.raises(ValueError, match="must be greater than 0"):
        calculate_epi(annual_energy_kwh, 0.0)


def test_baseline_service_equipment_aggregation():
    service = BaselineService()
    eq1 = Equipment(
        equipment_id="EQ-1",
        equipment_type="HVAC",
        equipment_name="Chiller",
        rated_power_kw=20.0,
        quantity=1,
        hours_per_day=10.0,
        operating_days=20,
        utilization_factor=1.0,
        minimum_hours=5.0,
        maximum_hours=12.0,
        is_flexible=True,
    )
    eq2 = Equipment(
        equipment_id="EQ-2",
        equipment_type="Lighting",
        equipment_name="LEDs",
        rated_power_kw=5.0,
        quantity=2,
        hours_per_day=10.0,
        operating_days=20,
        utilization_factor=1.0,
        minimum_hours=8.0,
        maximum_hours=12.0,
        is_flexible=False,
    )
    res = service.compute_equipment_baseline([eq1, eq2])
    assert res["baseline_energy_kwh"] == 6000.0
    assert len(res["equipment_breakdown"]) == 2
