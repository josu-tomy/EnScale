"""
Tests for incentive matching service and incentive record schema validation.
"""

from domain.models import BuildingProfile, Equipment, IncentiveMatch
from domain.enums import EligibilityStatus
from services.incentive_service import IncentiveService
from services.validation_service import validate_incentive_record


def test_incentive_match_model():
    inc = IncentiveMatch(
        incentive_id="INC-01",
        program_name="State DSM Rebate",
        target_equipment_type="HVAC",
        eligibility_status=EligibilityStatus.ELIGIBLE.value,
        estimated_rebate_inr=10000.0,
        policy_reference="Ref 101",
        description="High efficiency rebate",
    )
    val = validate_incentive_record(inc)
    assert val.is_valid is True
    assert len(val.errors) == 0


def test_incentive_record_validation_invalid():
    invalid_data = {
        "incentive_id": "INC-02",
        "program_name": "Test Scheme",
        "estimated_rebate_inr": -500.0,
    }
    val = validate_incentive_record(invalid_data)
    assert val.is_valid is False
    assert any("Missing required incentive field" in err for err in val.errors)


def test_incentive_service_matching():
    service = IncentiveService()
    building = BuildingProfile(
        building_id="B1",
        building_type="Commercial Office",
        location_state="Delhi",
        climate_zone="Composite",
        net_built_up_area_m2=1500.0,
        operating_start=9,
        operating_end=18,
        operating_days_per_month=22,
        occupancy=80.0,
        monthly_budget_inr=100000.0,
        tariff_inr_per_kwh=9.0,
    )
    eq = Equipment(
        equipment_id="EQ-1",
        equipment_type="HVAC",
        equipment_name="Office AC",
        rated_power_kw=15.0,
        quantity=2,
        hours_per_day=9.0,
        operating_days=22,
        utilization_factor=0.8,
        minimum_hours=6.0,
        maximum_hours=10.0,
        is_flexible=True,
    )
    matches = service.match_incentives(building, [eq])
    assert len(matches) >= 1
    assert matches[0].target_equipment_type == "HVAC"
