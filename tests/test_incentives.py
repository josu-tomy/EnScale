"""
Tests for incentive matching service, curated reference database, and payback estimation.

Verifies:
- Incentives reference JSON schema completeness
- find_incentives function filtering by geography, sector, and equipment
- Strict terminology safeguards: "Potential incentive match" and "Verify eligibility before application."
- Never emits "you qualify"
- Equipment upgrade modeled payback calculation and fallback handling
"""

import json
import pytest
from pathlib import Path

from domain.models import BuildingProfile, Equipment, IncentiveMatch
from domain.enums import EligibilityStatus
from services.incentive_service import (
    IncentiveService,
    find_incentives,
    calculate_equipment_upgrade_payback,
    load_incentive_database,
)
from services.validation_service import validate_incentive_record
from config.settings import REFERENCE_DATA_DIR


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


def test_incentives_json_schema_compliance():
    """Verifies all curated incentive records contain the 13 required canonical fields."""
    db = load_incentive_database()
    assert len(db) >= 6, "Expected at least 6 curated Indian efficiency incentive records"

    required_keys = [
        "incentive_id",
        "name",
        "authority",
        "region",
        "eligible_entity",
        "applicable_sector",
        "technology",
        "eligibility",
        "incentive_type",
        "incentive_value",
        "validity_period",
        "source",
        "last_verified",
    ]

    for record in db:
        for k in required_keys:
            assert k in record, f"Missing required key '{k}' in incentive record {record.get('incentive_id')}"
            assert len(str(record[k]).strip()) > 0, f"Empty value for '{k}' in {record.get('incentive_id')}"


def test_find_incentives_national_and_state_filtering():
    """Verifies geographic and technology filtering."""
    eq_hvac = Equipment(
        equipment_id="EQ-CHILL-01",
        equipment_type="HVAC",
        equipment_name="Water Chiller",
        rated_power_kw=50.0,
        quantity=1,
        hours_per_day=10.0,
        operating_days=22,
        utilization_factor=0.8,
        minimum_hours=6.0,
        maximum_hours=10.0,
        is_flexible=True,
    )

    # 1. Facility in Maharashtra with HVAC
    matches_maha = find_incentives(
        location_state="Maharashtra",
        building_type="Commercial Office",
        equipment_list=[eq_hvac],
    )
    assert len(matches_maha) >= 2
    matched_ids_maha = [m["incentive"]["incentive_id"] for m in matches_maha]
    # Should include National BEE / EESL and Maharashtra DSM
    assert "INC-BEE-SL-01" in matched_ids_maha
    assert "INC-MAHA-DSM-04" in matched_ids_maha
    # Should NOT include Gujarat DSM
    assert "INC-GUJ-DSM-05" not in matched_ids_maha

    # 2. Facility in Gujarat with HVAC
    matches_guj = find_incentives(
        location_state="Gujarat",
        building_type="Commercial Office",
        equipment_list=[eq_hvac],
    )
    matched_ids_guj = [m["incentive"]["incentive_id"] for m in matches_guj]
    assert "INC-GUJ-DSM-05" in matched_ids_guj
    assert "INC-MAHA-DSM-04" not in matched_ids_guj


def test_find_incentives_terminology_safeguards():
    """
    Enforces that:
    1. 'you qualify' is NEVER returned.
    2. 'Potential incentive match' and 'Verify eligibility before application.' are used.
    """
    eq_pump = Equipment(
        equipment_id="EQ-PUMP-01",
        equipment_type="pump",
        equipment_name="Water Pump",
        rated_power_kw=15.0,
        quantity=2,
        hours_per_day=8.0,
        operating_days=22,
        utilization_factor=0.85,
        minimum_hours=4.0,
        maximum_hours=8.0,
        is_flexible=True,
    )

    matches = find_incentives(
        location_state="Karnataka",
        building_type="Small Industrial",
        equipment_list=[eq_pump],
    )
    assert len(matches) > 0

    for m in matches:
        # Check reasons and notes
        notes = m["eligibility_notes"]
        label = m["status_label"]
        all_text = (json.dumps(m)).lower()

        assert "you qualify" not in all_text, "Forbidden phrase 'you qualify' found in match output"
        assert "potential incentive match" in notes.lower()
        assert "verify eligibility before application" in notes.lower()
        assert label == "Potential incentive match"


def test_calculate_equipment_upgrade_payback_valid():
    """Verifies formula: modeled_payback = effective_investment / annual_cost_savings."""
    eq = Equipment(
        equipment_id="EQ-CHILL-01",
        equipment_type="HVAC",
        equipment_name="Central Chiller",
        rated_power_kw=40.0,
        quantity=1,
        hours_per_day=10.0,
        operating_days=22,
        utilization_factor=0.8,
        minimum_hours=6.0,
        maximum_hours=10.0,
        is_flexible=True,
    )

    # 40 kW * 1 unit * 35,000 INR/kW benchmark = 1,400,000 INR investment
    # 15% incentive = 210,000 INR
    # Effective investment = 1,190,000 INR
    # Annual savings = 238,000 INR
    # Modeled payback = 1,190,000 / 238,000 = 5.0 years
    res = calculate_equipment_upgrade_payback(
        equipment=eq,
        annual_cost_savings_inr=238000.0,
        potential_rebate_pct=0.15,
        benchmark_capex_per_kw=35000.0,
    )

    assert res["status"] == "Calculated"
    assert res["estimated_investment_inr"] == 1400000.0
    assert res["potential_incentive_inr"] == 210000.0
    assert res["effective_modeled_investment_inr"] == 1190000.0
    assert res["annual_modeled_savings_inr"] == 238000.0
    assert res["modeled_payback_years"] == 5.0


def test_calculate_equipment_upgrade_payback_insufficient_data():
    """Verifies graceful handling of non-positive savings or missing capacity."""
    eq_zero = Equipment(
        equipment_id="EQ-0",
        equipment_type="Misc",
        equipment_name="Sensor",
        rated_power_kw=0.0,
        quantity=1,
        hours_per_day=0.0,
        operating_days=0,
        utilization_factor=0.0,
        minimum_hours=0.0,
        maximum_hours=0.0,
        is_flexible=False,
    )

    res = calculate_equipment_upgrade_payback(
        equipment=eq_zero,
        annual_cost_savings_inr=50000.0,
    )
    assert res["status"] == "Payback unavailable."
    assert res["modeled_payback_years"] is None

    # Test with non-positive savings
    eq_valid = Equipment(
        equipment_id="EQ-1",
        equipment_type="HVAC",
        equipment_name="AC",
        rated_power_kw=10.0,
        quantity=1,
        hours_per_day=8.0,
        operating_days=20,
        utilization_factor=0.8,
        minimum_hours=4.0,
        maximum_hours=8.0,
        is_flexible=True,
    )
    res_zero_savings = calculate_equipment_upgrade_payback(
        equipment=eq_valid,
        annual_cost_savings_inr=0.0,
    )
    assert res_zero_savings["status"] == "Payback unavailable."
    assert res_zero_savings["modeled_payback_years"] is None


def test_find_incentives_technology_filtering_mismatch():
    """Verifies that technology mismatch properly excludes non-applicable programs."""
    eq_lighting = Equipment(
        equipment_id="EQ-LIGHT-01",
        equipment_type="Lighting",
        equipment_name="LED Office Lights",
        rated_power_kw=10.0,
        quantity=1,
        hours_per_day=12.0,
        operating_days=22,
        utilization_factor=1.0,
        minimum_hours=8.0,
        maximum_hours=12.0,
        is_flexible=False,
    )
    matches = find_incentives(
        location_state="Delhi",
        building_type="Commercial Office",
        equipment_list=[eq_lighting],
    )
    matched_ids = [m["incentive"]["incentive_id"] for m in matches]
    # Specialized chiller program (INC-EESL-CHILL-03) should NOT match lighting
    assert "INC-EESL-CHILL-03" not in matched_ids
    # SIDBI 4E (Motors/compressors/boilers) should NOT match commercial lighting
    assert "INC-SIDBI-4E-02" not in matched_ids
    # Delhi DSM rebate (INC-DEL-DSM-07) covers lighting and should match
    assert "INC-DEL-DSM-07" in matched_ids
