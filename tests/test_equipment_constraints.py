import json

import pandas as pd
import pytest

from config.constants import CANONICAL_EQUIPMENT_FIELDS
from domain.models import BuildingProfile, Equipment
from services.ai.manager import AIManager
from services.analysis_service import analyze_energy
from services.baseline_service import calculate_equipment_energy
from services.equipment_service import (
    add_equipment,
    create_equipment,
    equipment_canonical_fields,
    remove_equipment,
    validate_equipment_for_building,
)
from services.opportunity_service import identify_energy_opportunities
from services.optimization_service import (
    InfeasibleConstraintError,
    optimize_equipment_schedule,
    optimize_equipment_schedule_safely,
)


@pytest.fixture
def profile():
    return BuildingProfile("B", "Commercial Office", "Maharashtra", "Composite", 1000,
                           8, 16, 22, 20, 10000, 8.5)


def equipment(equipment_id="EQ-1", **overrides):
    values = dict(equipment_id=equipment_id, equipment_type="HVAC", equipment_name=equipment_id,
                  rated_power_kw=10.0, quantity=1, hours_per_day=6.0, operating_days=22,
                  utilization_factor=0.8, minimum_hours=4.0, maximum_hours=8.0, is_flexible=True)
    values.update(overrides)
    return Equipment(**values)


def test_minimum_runtime_over_window_is_invalid_and_direct_optimizer_keeps_domain_error(profile):
    item = equipment(minimum_hours=9, maximum_hours=9)
    validation = validate_equipment_for_building(item, profile)
    assert not validation.is_valid
    assert any("minimum_hours" in message and "operating window" in message for message in validation.errors)
    with pytest.raises(InfeasibleConstraintError, match="minimum runtime.*exceeds building operating window"):
        optimize_equipment_schedule(profile, [item])


def test_maximum_runtime_over_window_is_invalid(profile):
    item = equipment(maximum_hours=9)
    result = validate_equipment_for_building(item, profile)
    assert not result.is_valid
    assert any("maximum_hours" in message for message in result.errors)


def test_daily_runtime_over_window_is_invalid(profile):
    result = validate_equipment_for_building(equipment(hours_per_day=9), profile)
    assert not result.is_valid
    assert any("hours_per_day" in message for message in result.errors)


def test_valid_equipment_optimizes(profile):
    result = optimize_equipment_schedule_safely(profile, [equipment()])
    assert result.is_feasible
    assert len(result.schedule_recommendations) == 1


def test_no_flexible_equipment_is_a_valid_zero_savings_result(profile):
    item = equipment(is_flexible=False, hours_per_day=8, minimum_hours=8, maximum_hours=8)
    result = optimize_equipment_schedule_safely(profile, [item])
    assert result.is_feasible
    assert result.energy_savings_kwh == 0
    assert result.schedule_recommendations[0]["equipment_id"] == item.equipment_id


def test_multiple_equipment_skips_only_infeasible_item(profile):
    bad = equipment("BAD", minimum_hours=9, maximum_hours=9)
    good = equipment("GOOD", equipment_name="Pump")
    warnings = []
    opportunities = identify_energy_opportunities(None, profile, [bad, good], warnings=warnings)
    assert any(o.equipment_id == "GOOD" for o in opportunities)
    assert not any(o.equipment_id == "BAD" for o in opportunities)
    assert warnings == ["BAD could not be optimized because its minimum runtime exceeds the building operating window. Review its operating hours."]


def test_all_equipment_infeasible_returns_no_feasible_optimization(profile):
    result = optimize_equipment_schedule_safely(profile, [
        equipment("BAD-1", minimum_hours=9, maximum_hours=9),
        equipment("BAD-2", maximum_hours=9),
    ])
    assert not result.is_feasible
    assert "No feasible optimization" in result.status_message
    assert result.schedule_recommendations == []
    assert result.energy_savings_kwh == 0
    assert len(result.warnings) == 2


def test_imported_equipment_dict_is_validated_before_model_construction(profile):
    raw = equipment("IMPORTED", minimum_hours=9, maximum_hours=9).to_dict()
    result = optimize_equipment_schedule_safely(profile, [raw])
    assert not result.is_feasible
    assert "minimum runtime exceeds the building operating window" in result.warnings[0]


def test_impossible_constraints_do_not_crash_end_to_end_analysis(profile):
    df = pd.read_csv("data/demo/demo_office_energy.csv", nrows=72, parse_dates=["timestamp"])
    bad = equipment("BAD", minimum_hours=9, maximum_hours=9)
    result = analyze_energy(profile, [bad], df, ai_manager=AIManager("deterministic"))
    assert result.optimization["is_feasible"] is False
    assert result.optimization["schedule_recommendations"] == []
    assert result.optimization["warnings"]
    assert all(stage["status"] == "complete" for stage in result.progress)


def test_quantity_is_persisted_and_changes_modeled_energy():
    one = equipment(quantity=1)
    two = Equipment.from_dict({**one.to_dict(), "quantity": 2})
    assert two.to_dict()["quantity"] == 2
    assert calculate_equipment_energy([two])["total_energy"] == 2 * calculate_equipment_energy([one])["total_energy"]


def test_equipment_collection_supports_add_and_remove(profile):
    inventory = []
    item = create_equipment("NEW-1", profile.operating_end - profile.operating_start)
    add_equipment(inventory, item)
    assert inventory == [item]
    assert remove_equipment(inventory, "NEW-1")
    assert inventory == []


def test_building_editor_add_remove_and_persists_quantity():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(Path(__file__).resolve().parent.parent / "app.py").run(timeout=15)
    original_count = len(at.session_state["equipment_inventory"])
    at.button(key="add_equipment").click().run(timeout=15)
    inventory = at.session_state["equipment_inventory"]
    assert len(inventory) == original_count + 1
    created = inventory[-1]

    at.number_input(key=f"equipment_{created.equipment_id}_quantity").set_value(3).run(timeout=15)
    inventory = at.session_state["equipment_inventory"]
    assert next(item for item in inventory if item.equipment_id == created.equipment_id).quantity == 3

    at.button(key=f"equipment_{created.equipment_id}_remove").click().run(timeout=15)
    assert len(at.session_state["equipment_inventory"]) == original_count
    assert all(item.equipment_id != created.equipment_id for item in at.session_state["equipment_inventory"])


def test_understand_screen_warns_and_continues_with_infeasible_equipment():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(Path(__file__).resolve().parent.parent / "app.py").run(timeout=15)
    profile = at.session_state["building_profile"]
    profile.operating_end = 16
    at.session_state["building_profile"] = profile
    light = next(item for item in at.session_state["equipment_inventory"] if item.equipment_id == "EQ-LIGHT-01")
    light.hours_per_day = 10
    light.minimum_hours = 10
    light.maximum_hours = 10
    at.session_state["workflow_step"] = "3. Understand"
    at.run(timeout=20)

    assert not at.exception
    warnings = [item.value for item in at.warning]
    assert any("Workstation & Floor Lighting could not be optimized because its minimum runtime exceeds the building operating window" in warning for warning in warnings)


def test_building_editor_revalidates_after_operating_hours_change():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(Path(__file__).resolve().parent.parent / "app.py").run(timeout=15)
    at.get_by_key("equipment_EQ-LIGHT-01_hours_per_day").set_value(10).run(timeout=15)
    at.get_by_key("equipment_EQ-LIGHT-01_minimum_hours").set_value(8).run(timeout=15)
    at.get_by_key("equipment_EQ-LIGHT-01_maximum_hours").set_value(10).run(timeout=15)
    at.get_by_key("building_operating_end").set_value(16).run(timeout=15)
    assert any("maximum_hours" in item.value and "operating window" in item.value for item in at.error)
    unchanged_light = next(item for item in at.session_state["equipment_inventory"] if item.equipment_id == "EQ-LIGHT-01")
    assert unchanged_light.maximum_hours == 10


def test_runtime_validation_rechecks_when_building_window_changes(profile):
    item = equipment(maximum_hours=8)
    assert validate_equipment_for_building(item, profile).is_valid
    shorter = BuildingProfile.from_dict({**profile.to_dict(), "operating_end": 15})
    result = validate_equipment_for_building(item, shorter)
    assert not result.is_valid
    assert any("maximum_hours" in message for message in result.errors)


def test_canonical_equipment_fields_remain_unchanged():
    expected = ("equipment_id", "equipment_type", "equipment_name", "rated_power_kw", "quantity",
                "hours_per_day", "operating_days", "utilization_factor", "minimum_hours",
                "maximum_hours", "is_flexible")
    assert CANONICAL_EQUIPMENT_FIELDS == expected
    assert equipment_canonical_fields() == expected


def test_demo_equipment_fits_eight_hour_window():
    with open("data/demo/demo_equipment.json", encoding="utf-8") as source:
        inventory = json.load(source)
    for item in inventory:
        assert item["hours_per_day"] <= 8
        assert item["minimum_hours"] <= 8
        assert item["maximum_hours"] <= 8
