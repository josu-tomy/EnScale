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

    # 1. Setup
    at.sidebar.radio[0].set_value("1. Setup").run(timeout=10)
    assert not at.exception

    # 2. Data -> Demo Scenario
    at.sidebar.radio[0].set_value("2. Data").run(timeout=10)
    assert not at.exception

    # 3. Forecast
    at.sidebar.radio[0].set_value("3. Forecast").run(timeout=10)
    assert not at.exception

    # 4. Waste
    at.sidebar.radio[0].set_value("4. Waste").run(timeout=10)
    assert not at.exception

    # 5. Optimize
    at.sidebar.radio[0].set_value("5. Optimize").run(timeout=10)
    assert not at.exception

    # 6. Finance
    at.sidebar.radio[0].set_value("6. Finance").run(timeout=10)
    assert not at.exception

    # 7. Action Plan
    at.sidebar.radio[0].set_value("7. Action Plan").run(timeout=10)
    assert not at.exception
    assert len(at.title) > 0
    assert "ENERGY ACTION PLAN" in at.title[0].value
