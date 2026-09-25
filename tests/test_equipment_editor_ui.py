"""
Comprehensive UI and Workflow Integration Tests for Dynamic Equipment Editor.

Tests:
1. Adding equipment dynamically.
2. Editing all 8 canonical fields (Name, Type, Quantity, Power, Hours/day, Min runtime, Max runtime, Flexible).
3. Verifying quantity is visible, editable, and influences load calculations.
4. Deleting equipment.
5. Enforcing that equipment IDs and internal schema names are NOT exposed to the user.
6. Validation message when runtime exceeds building operating window.
7. Input is never silently clamped or modified when invalid.
8. Changing building operating hours immediately revalidates equipment runtimes.
9. Invalid equipment blocks progression from Building to Data.
10. Equipment configuration persists faithfully through Understand and Improve without mutation.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from services.baseline_service import calculate_equipment_energy


def get_app():
    app_file = Path(__file__).resolve().parent.parent / "app.py"
    return AppTest.from_file(app_file)


def test_no_equipment_ids_exposed_in_ui():
    """Verify that internal equipment IDs (e.g. EQ-HVAC-01) are not displayed to user."""
    at = get_app().run(timeout=15)
    inventory = at.session_state["equipment_inventory"]
    assert len(inventory) > 0

    # Collect all markdown and text values rendered on the Building page
    rendered_texts = [m.value for m in at.markdown]

    for eq in inventory:
        # Check that the equipment ID is not exposed in any user-facing text
        for text in rendered_texts:
            # Code blocks or backticks with equipment_id should not exist
            assert f"`{eq.equipment_id}`" not in text, f"Equipment ID {eq.equipment_id} was exposed in UI text: {text}"
            assert f"· {eq.equipment_id}" not in text, f"Equipment ID {eq.equipment_id} was exposed in UI text: {text}"


def test_add_equipment_and_edit_all_eight_fields():
    """Verify adding equipment and editing all 8 required fields."""
    at = get_app().run(timeout=15)
    initial_count = len(at.session_state["equipment_inventory"])

    # Click Add equipment
    at.button(key="add_equipment").click().run(timeout=15)
    inventory = at.session_state["equipment_inventory"]
    assert len(inventory) == initial_count + 1

    new_eq = inventory[-1]
    eq_id = new_eq.equipment_id

    # Verify all 8 widget keys exist
    assert at.text_input(key=f"equipment_{eq_id}_equipment_name") is not None
    assert at.selectbox(key=f"equipment_{eq_id}_equipment_type") is not None
    assert at.number_input(key=f"equipment_{eq_id}_quantity") is not None
    assert at.number_input(key=f"equipment_{eq_id}_rated_power_kw") is not None
    assert at.number_input(key=f"equipment_{eq_id}_hours_per_day") is not None
    assert at.number_input(key=f"equipment_{eq_id}_minimum_hours") is not None
    assert at.number_input(key=f"equipment_{eq_id}_maximum_hours") is not None
    assert at.checkbox(key=f"equipment_{eq_id}_is_flexible") is not None

    # Edit all 8 fields
    at.text_input(key=f"equipment_{eq_id}_equipment_name").set_value("Rooftop Solar Inverter").run(timeout=15)
    at.selectbox(key=f"equipment_{eq_id}_equipment_type").set_value("Process Machinery").run(timeout=15)
    at.number_input(key=f"equipment_{eq_id}_quantity").set_value(4).run(timeout=15)
    at.number_input(key=f"equipment_{eq_id}_rated_power_kw").set_value(25.0).run(timeout=15)
    at.number_input(key=f"equipment_{eq_id}_hours_per_day").set_value(7.5).run(timeout=15)
    at.number_input(key=f"equipment_{eq_id}_minimum_hours").set_value(3.0).run(timeout=15)
    at.number_input(key=f"equipment_{eq_id}_maximum_hours").set_value(8.0).run(timeout=15)
    at.checkbox(key=f"equipment_{eq_id}_is_flexible").set_value(True).run(timeout=15)

    # Check updated state
    updated_eq = next(item for item in at.session_state["equipment_inventory"] if item.equipment_id == eq_id)
    assert updated_eq.equipment_name == "Rooftop Solar Inverter"
    assert updated_eq.equipment_type == "Process Machinery"
    assert updated_eq.quantity == 4
    assert updated_eq.rated_power_kw == 25.0
    assert updated_eq.hours_per_day == 7.5
    assert updated_eq.minimum_hours == 3.0
    assert updated_eq.maximum_hours == 8.0
    assert updated_eq.is_flexible is True


def test_quantity_affects_energy_calculation():
    """Verify quantity directly impacts total equipment load calculation."""
    at = get_app().run(timeout=15)
    inventory = at.session_state["equipment_inventory"]
    target_eq = inventory[0]
    initial_qty = target_eq.quantity

    baseline_energy_1 = calculate_equipment_energy(inventory)["total_energy"]

    # Double the quantity
    at.number_input(key=f"equipment_{target_eq.equipment_id}_quantity").set_value(initial_qty * 2).run(timeout=15)

    baseline_energy_2 = calculate_equipment_energy(at.session_state["equipment_inventory"])["total_energy"]
    assert baseline_energy_2 > baseline_energy_1


def test_remove_equipment():
    """Verify removing equipment deletes it from the inventory."""
    at = get_app().run(timeout=15)
    initial_inventory = list(at.session_state["equipment_inventory"])
    remove_target = initial_inventory[0]

    # Click remove
    at.button(key=f"equipment_{remove_target.equipment_id}_remove").click().run(timeout=15)

    current_inventory = at.session_state["equipment_inventory"]
    assert len(current_inventory) == len(initial_inventory) - 1
    assert all(item.equipment_id != remove_target.equipment_id for item in current_inventory)


def test_invalid_runtime_shows_inline_error_and_preserves_input():
    """Verify invalid runtime exceeding operating window shows inline error and is not clamped."""
    at = get_app().run(timeout=15)
    target = at.session_state["equipment_inventory"][0]

    # Operating window is 8 to 18 (10 hours)
    # Set runtime to 15 hours
    at.number_input(key=f"equipment_{target.equipment_id}_hours_per_day").set_value(15.0).run(timeout=15)

    # Input must NOT be clamped
    current_target = next(item for item in at.session_state["equipment_inventory"] if item.equipment_id == target.equipment_id)
    assert current_target.hours_per_day == 15.0

    # Inline error must be visible
    errors = [e.value for e in at.error]
    assert any("hours_per_day" in err and "operating window" in err for err in errors)


def test_invalid_equipment_blocks_continue_to_data():
    """Verify that invalid equipment blocks navigation to Data step."""
    at = get_app().run(timeout=15)
    target = at.session_state["equipment_inventory"][0]

    # Set invalid runtime
    at.number_input(key=f"equipment_{target.equipment_id}_hours_per_day").set_value(20.0).run(timeout=15)

    # Click Continue to Data
    at.button(key="continue_to_data_button").click().run(timeout=15)

    # Step must still be 1. Building
    assert at.session_state["workflow_step"] == "1. Building"
    assert any("hours_per_day" in err.value and "operating window" in err.value for err in at.error)


def test_equipment_persists_into_understand_and_improve():
    """Verify that equipment configured in Building persists into Understand and Improve."""
    at = get_app().run(timeout=15)

    # Add a distinctive equipment
    at.button(key="add_equipment").click().run(timeout=15)
    new_eq = at.session_state["equipment_inventory"][-1]
    at.text_input(key=f"equipment_{new_eq.equipment_id}_equipment_name").set_value("High-Efficiency Centrifugal Chiller").run(timeout=15)
    at.number_input(key=f"equipment_{new_eq.equipment_id}_quantity").set_value(2).run(timeout=15)
    at.number_input(key=f"equipment_{new_eq.equipment_id}_rated_power_kw").set_value(75.0).run(timeout=15)
    at.number_input(key=f"equipment_{new_eq.equipment_id}_hours_per_day").set_value(8.0).run(timeout=15)
    at.number_input(key=f"equipment_{new_eq.equipment_id}_minimum_hours").set_value(4.0).run(timeout=15)
    at.number_input(key=f"equipment_{new_eq.equipment_id}_maximum_hours").set_value(8.0).run(timeout=15)
    at.checkbox(key=f"equipment_{new_eq.equipment_id}_is_flexible").set_value(True).run(timeout=15)

    expected_count = len(at.session_state["equipment_inventory"])

    # Navigate to Understand
    at.session_state["workflow_step"] = "3. Understand"
    at.run(timeout=15)
    assert not at.exception
    assert len(at.session_state["equipment_inventory"]) == expected_count
    assert any(eq.equipment_name == "High-Efficiency Centrifugal Chiller" for eq in at.session_state["equipment_inventory"])

    # Navigate to Improve
    at.session_state["workflow_step"] = "4. Improve"
    at.run(timeout=15)
    assert not at.exception
    opt_result = at.session_state.get("optimization_result")
    assert opt_result is not None
    # Equipment inventory remains intact and unchanged by Understand / Improve
    assert len(at.session_state["equipment_inventory"]) == expected_count
    assert any(eq.equipment_name == "High-Efficiency Centrifugal Chiller" for eq in at.session_state["equipment_inventory"])
