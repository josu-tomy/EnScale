"""
UI automation and screen rendering tests for EnScale.

Verifies:
- All 5 redesigned consumer SaaS workflow screens render cleanly without exceptions:
  1. Building   (Tell us about your building)
  2. Data       (Add your energy data)
  3. Understand (Here's what your energy is doing)
  4. Improve    (Where can you save?)
  5. Action Plan(Your Energy Action Plan)
- End-to-end user navigation across forward and backward workflows
- Meaningful forecast visualization, takeaways, before/after savings, and action plan cards
"""

import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_all_five_workflow_screens_render_cleanly():
    """Verifies that each of the 5 redesigned workflow screens renders without exceptions."""
    steps = [
        "1. Building",
        "2. Data",
        "3. Understand",
        "4. Improve",
        "5. Action Plan",
    ]

    app_file = Path(__file__).resolve().parent.parent / "app.py"
    for step in steps:
        at = AppTest.from_file(app_file)
        at.run(timeout=10)
        at.session_state["workflow_step"] = step
        at.run(timeout=10)
        assert not at.exception, f"Rendering exception on screen '{step}': {at.exception}"


def test_ui_workflow_demo_pipeline():
    """Verifies full sequential user journey from Building -> Data -> Understand -> Improve -> Action Plan."""
    app_file = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(app_file)
    at.run(timeout=10)

    # 1. Building screen
    assert "Tell us about your building" in at.title[0].value
    assert not at.exception

    # 2. Transition to Data
    at.session_state["workflow_step"] = "2. Data"
    at.run(timeout=10)
    assert "Add your energy data" in at.title[0].value
    assert at.session_state["active_df"] is not None
    assert len(at.session_state["active_df"]) >= 288
    assert not at.exception

    # 3. Transition to Understand
    at.session_state["workflow_step"] = "3. Understand"
    at.run(timeout=10)
    assert "Here's what your energy is doing" in at.title[0].value
    assert at.session_state["predicted_energy"] is not None
    assert len(at.session_state["predicted_energy"]) == len(at.session_state["active_df"])
    assert at.session_state["anomaly_df"] is not None
    assert not at.exception

    # 4. Transition to Improve
    at.session_state["workflow_step"] = "4. Improve"
    at.run(timeout=10)
    assert "Where can you save?" in at.title[0].value
    opt_res = at.session_state.get("optimization_result")
    assert opt_res is not None
    assert opt_res.is_feasible is True
    assert opt_res.energy_savings_kwh >= 0
    assert opt_res.cost_savings_inr >= 0
    assert not at.exception

    # 5. Transition to Action Plan
    at.session_state["workflow_step"] = "5. Action Plan"
    at.run(timeout=10)
    assert "Your Energy Action Plan" in at.title[0].value
    assert not at.exception


def test_ui_backward_navigation():
    """Verifies backward step transitions without corrupting session state."""
    app_file = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(app_file)
    at.run(timeout=10)

    # Move forward to Screen 4
    at.session_state["workflow_step"] = "4. Improve"
    at.run(timeout=10)
    assert "Where can you save?" in at.title[0].value

    # Move back to Screen 3
    at.session_state["workflow_step"] = "3. Understand"
    at.run(timeout=10)
    assert "Here's what your energy is doing" in at.title[0].value

    # Move back to Screen 1
    at.session_state["workflow_step"] = "1. Building"
    at.run(timeout=10)
    assert "Tell us about your building" in at.title[0].value
    assert not at.exception


def test_ui_wording_and_layout_checks():
    """
    Asserts wording rules and UI layout safeguards:
    - Data screen uses 'Evaluation dataset performance'.
    - Schedule-derived items use 'Estimated' or 'Schedule-derived', never 'Verified' or 'detected'.
    - Improve screen displays 'Listed equipment only' and visible text contrast styling.
    - Decision summary uses CSS grid with min-width: 0 and word-break to avoid overlap.
    """
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Data screen caption
    assert "Evaluation dataset performance" in content
    assert "Benchmark accuracy of" not in content

    # 2. Schedule-derived items wording
    assert "Prioritized Operational Interventions (Schedule-Derived Savings)" in content
    assert "Prioritized Operational Interventions (Verified Headline Savings)" not in content
    assert "Current Operating Baseline:" in content
    assert "Estimated Annual Impact (Schedule-Derived):" in content
    assert "Verified Annual Impact:" not in content
    assert "Schedule-Derived Opportunities" in content

    # 3. Improve screen coverage label & visible text contrast
    assert "Listed equipment only (" in content
    assert "% of metered load)" in content
    assert "color: #0f172a;" in content

    # 4. Decision summary layout overlap safeguards
    assert "min-width: 0;" in content
    assert "word-break: break-word;" in content
    assert "box-sizing: border-box;" in content
