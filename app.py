"""
EnScale - ML-assisted energy decision platform for commercial and small industrial buildings.

Streamlit application entrypoint (Foundation / Skeleton).
"""

import streamlit as st

from config.constants import APP_NAME, APP_TAGLINE, APP_VERSION
from config.settings import settings
from domain.enums import DataMode


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} - Energy Decision Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar Navigation / Stage Info
    st.sidebar.title(f"⚡ {APP_NAME}")
    st.sidebar.caption(f"v{APP_VERSION} (Foundation Stage)")
    st.sidebar.info("Local-First Energy Decision Platform")

    mode = st.sidebar.radio(
        "Operating Mode",
        options=[
            DataMode.MODE_1_HISTORICAL.value,
            DataMode.MODE_2_EQUIPMENT_BASELINE.value,
            DataMode.MODE_3_DEMO.value,
        ],
        format_func=lambda x: {
            DataMode.MODE_1_HISTORICAL.value: "Mode 1: Historical Data",
            DataMode.MODE_2_EQUIPMENT_BASELINE.value: "Mode 2: Equipment Baseline",
            DataMode.MODE_3_DEMO.value: "Mode 3: Demo Scenario",
        }.get(x, x),
        index=2,
    )

    # Main Header
    st.title(APP_NAME)
    st.write(f"*{APP_TAGLINE}*")

    st.divider()

    # Core Pipeline Sections / Placeholders
    with st.expander("1. Setup", expanded=True):
        st.subheader("Building & Operational Setup")
        st.write("Configure building profile, location, operating hours, and baseline tariff.")
        st.info("Setup module ready for Stage 2 integration.")

    with st.expander("2. Data", expanded=False):
        st.subheader("Data Ingestion & Verification")
        st.write("Ingest historical 15-min / hourly energy CSV or configure equipment inventories.")
        st.caption(f"Current active mode: {mode}")

    with st.expander("3. Forecast", expanded=False):
        st.subheader("ML Energy Forecast")
        st.write("Predict expected baseline energy consumption using ML regression models.")

    with st.expander("4. Waste", expanded=False):
        st.subheader("Anomaly & Waste Detection")
        st.write("Identify operational waste, off-hour consumption, and unusual energy spikes.")

    with st.expander("5. Optimize", expanded=False):
        st.subheader("Constrained Energy Optimization")
        st.write("Simulate equipment rescheduling, setpoint adjustments, and peak-load shaving.")

    with st.expander("6. Finance", expanded=False):
        st.subheader("Financial Impact & Modeled Savings")
        st.write("Quantify modeled ₹ cost savings, kWh reductions, and matched incentive policies.")

    with st.expander("7. Action Plan", expanded=False):
        st.subheader("Final Energy Action Plan")
        st.write("Comprehensive summary of actionable energy interventions with modeled ROI.")


if __name__ == "__main__":
    main()
