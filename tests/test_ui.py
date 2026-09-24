"""
UI automation and screen rendering tests for EnScale.

Verifies:
- All 7 screens render cleanly without exceptions via Streamlit AppTest:
  1. Setup
  2. Data
  3. Forecast
  4. Waste
  5. Optimize
  6. Finance
  7. Action Plan
- End-to-end user navigation across workflows
- Data source badge presence
"""

import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_all_seven_pipeline_screens_render_cleanly():
    """Verifies that each of the 7 pipeline screens renders without exceptions."""
    steps = [
        "1. Setup",
        "2. Data",
        "3. Forecast",
        "4. Waste",
        "5. Optimize",
        "6. Finance",
        "7. Action Plan",
    ]

    app_file = Path(__file__).resolve().parent.parent / "app.py"
    for step in steps:
        at = AppTest.from_file(app_file)
        at.run(timeout=10)
        at.sidebar.radio[0].set_value(step).run(timeout=10)
        assert not at.exception, f"Rendering exception on screen '{step}': {at.exception}"


def test_ui_workflow_demo_pipeline():
    """Verifies full Demo workflow through Setup -> Data -> Forecast -> Waste -> Optimize -> Finance -> Action Plan."""
    app_file = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(app_file)
    at.run(timeout=10)

    # Step 1: Launch app & Step 2: Setup
    at.sidebar.radio[0].set_value("1. Setup").run(timeout=10)
    assert not at.exception
    assert "Facility & Operational Setup" in at.title[0].value

    # Step 3: Data -> Demo Scenario & Step 4: Summary stats
    at.sidebar.radio[0].set_value("2. Data").run(timeout=10)
    assert not at.exception
    # Ensure active demo dataset has >= 288 rows (12 days hourly)
    assert at.session_state["active_df"] is not None
    assert len(at.session_state["active_df"]) >= 288

    # Step 5: Forecast -> load curve and metrics
    at.sidebar.radio[0].set_value("3. Forecast").run(timeout=10)
    assert not at.exception
    assert at.session_state["predicted_energy"] is not None
    assert len(at.session_state["predicted_energy"]) == len(at.session_state["active_df"])

    # Step 6: Waste -> identify anomalies, severity, and cost
    at.sidebar.radio[0].set_value("4. Waste").run(timeout=10)
    assert not at.exception
    assert at.session_state["anomaly_df"] is not None

    # Step 7: Optimize -> run constrained optimization & Step 8: check savings
    at.sidebar.radio[0].set_value("5. Optimize").run(timeout=10)
    assert not at.exception
    opt_res = at.session_state.get("optimization_result")
    assert opt_res is not None
    assert opt_res.is_feasible is True
    assert opt_res.energy_savings_kwh >= 0
    assert opt_res.cost_savings_inr >= 0

    # Step 9: Finance -> matched incentives, eligibility notes, modeled payback
    at.sidebar.radio[0].set_value("6. Finance").run(timeout=10)
    assert not at.exception
    matches = at.session_state.get("matched_incentives", [])
    assert len(matches) > 0
    for m in matches:
        assert "you qualify" not in str(m).lower()

    # Step 10: Action Plan -> 8 sections populate
    at.sidebar.radio[0].set_value("7. Action Plan").run(timeout=10)
    assert not at.exception
    assert len(at.title) > 0
    assert "ENERGY ACTION PLAN" in at.title[0].value
    # Step 11: Zero crashes throughout
    assert not at.exception


def test_ui_equipment_baseline_workflow():
    """Verifies Equipment Model UI workflow without interval data."""
    app_file = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(app_file)
    at.run(timeout=10)

    # 1. Setup
    at.sidebar.radio[0].set_value("1. Setup").run(timeout=10)
    assert not at.exception

    # 2. Data -> Select Build Baseline From Equipment
    at.sidebar.radio[0].set_value("2. Data").run(timeout=10)
    at.radio[0].set_value("Build Baseline From Equipment").run(timeout=10)
    assert not at.exception
    assert at.session_state["data_source_label"] == "Equipment model"

    # 3. Forecast should show graceful explanation
    at.sidebar.radio[0].set_value("3. Forecast").run(timeout=10)
    assert not at.exception
    assert any("Equipment model" in str(w.value) for w in at.warning)

    # 4. Waste should show graceful explanation
    at.sidebar.radio[0].set_value("4. Waste").run(timeout=10)
    assert not at.exception
    assert any("Equipment model" in str(w.value) for w in at.warning)

    # 5. Optimize works directly with equipment model
    at.sidebar.radio[0].set_value("5. Optimize").run(timeout=10)
    assert not at.exception

    # 6. Finance works
    at.sidebar.radio[0].set_value("6. Finance").run(timeout=10)
    assert not at.exception

    # 7. Action Plan generates clean executive report
    at.sidebar.radio[0].set_value("7. Action Plan").run(timeout=10)
    assert not at.exception
    assert "ENERGY ACTION PLAN" in at.title[0].value
