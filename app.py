"""
EnScale - ML-assisted energy decision platform for commercial and small industrial buildings.

Streamlit application entrypoint.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd

from config.constants import APP_NAME, APP_TAGLINE, APP_VERSION
from config.settings import settings, DEMO_DATA_DIR, REFERENCE_DATA_DIR
from domain.enums import DataMode
from services.ingestion_service import load_energy_csv, get_energy_csv_template, IngestionService
from services.baseline_service import calculate_equipment_energy, BaselineService


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} - Energy Decision Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    st.sidebar.title(f"⚡ {APP_NAME}")
    st.sidebar.caption(f"v{APP_VERSION} (Decision Platform)")
    st.sidebar.info("Local-First Energy Decision Support")

    mode_choice = st.sidebar.radio(
        "Operating Mode",
        options=[
            "Use My Historical Data",
            "Build Baseline From Equipment",
            "Demo Scenario",
        ],
        index=2,
    )

    # Header
    st.title(APP_NAME)
    st.markdown(f"*{APP_TAGLINE}*")

    st.divider()

    # Section 1: Setup Placeholder
    with st.expander("1. Setup", expanded=False):
        st.subheader("Building & Operational Setup")
        st.write("Configured building baseline profiles, location parameters, and utility tariffs.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Default Tariff", f"₹ {settings.default_tariff_inr_per_kwh:.2f} / kWh")
        col2.metric("Grid Emission Factor", f"{settings.default_emission_factor_kg_per_kwh:.2f} kg CO₂ / kWh")
        col3.metric("Platform Architecture", "100% Local")

    # Section 2: Data Screen (Interactive Implementation)
    with st.expander("2. Data (Ingestion & Baseline)", expanded=True):
        st.subheader(f"Data Workspace — {mode_choice}")

        if mode_choice == "Use My Historical Data":
            st.markdown("Upload interval energy meter data (15-minute, 30-minute, or hourly).")
            
            # Downloadable template
            template_csv = get_energy_csv_template()
            st.download_button(
                label="📥 Download CSV Template",
                data=template_csv,
                file_name="enscale_energy_template.csv",
                mime="text/csv",
                help="Template with canonical columns: timestamp, energy_kwh, temperature_c, occupancy, equipment_load_kw",
            )

            uploaded_file = st.file_uploader(
                "Upload Energy Consumption CSV",
                type=["csv"],
                help="Requires canonical columns: timestamp, energy_kwh",
            )

            if uploaded_file is not None:
                try:
                    df = load_energy_csv(uploaded_file)
                    val_info = df.attrs.get("validation_info", {})

                    st.success("✅ CSV Successfully Validated and Ingested")
                    
                    # Key metrics cards
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Rows", f"{val_info.get('row_count', len(df)):,}")
                    m2.metric("Sampling Frequency", val_info.get("sampling_frequency", "Unknown"))
                    m3.metric("Duplicate Timestamps", str(val_info.get("duplicate_timestamps_count", 0)))
                    m4.metric("Negative Energy", str(val_info.get("negative_energy_count", 0)))

                    st.markdown(
                        f"**Date Range:** `{val_info.get('date_range_start', 'N/A')}` to `{val_info.get('date_range_end', 'N/A')}`"
                    )

                    # Warnings if any
                    warnings = val_info.get("warnings", [])
                    if warnings:
                        with st.expander("⚠️ Data Quality Advisories", expanded=False):
                            for w in warnings:
                                st.warning(w)

                    # Preview table
                    st.subheader("Data Preview (Chronologically Sorted)")
                    st.dataframe(df.head(50), use_container_width=True)

                except ValueError as ve:
                    st.error(f"❌ Data Validation Error: {ve}")
                except Exception as e:
                    st.error(f"❌ Unexpected error loading CSV: {e}")

        elif mode_choice == "Build Baseline From Equipment":
            st.markdown(
                "Establish energy baselines from operational equipment inventories. "
                "Calculations use the canonical formula: "
                "`estimated_energy_kwh = rated_power_kw × quantity × hours_per_day × operating_days × utilization_factor`."
            )

            # Load default equipment from demo reference if available
            eq_json_path = DEMO_DATA_DIR / "demo_equipment.json"
            if eq_json_path.exists():
                with open(eq_json_path) as f:
                    equipment_inventory = json.load(f)
            else:
                equipment_inventory = [
                    {
                        "equipment_id": "EQ-01",
                        "equipment_type": "HVAC",
                        "equipment_name": "Primary Chiller",
                        "rated_power_kw": 45.0,
                        "quantity": 2,
                        "hours_per_day": 10.0,
                        "operating_days": 22,
                        "utilization_factor": 0.75,
                        "minimum_hours": 6.0,
                        "maximum_hours": 12.0,
                        "is_flexible": True,
                    }
                ]

            baseline_calc = calculate_equipment_energy(equipment_inventory)

            col1, col2 = st.columns(2)
            col1.metric("Total Modeled Energy Baseline", f"{baseline_calc['total_energy']:,.2f} kWh / month")
            col2.metric("Configured Equipment Assets", len(equipment_inventory))

            st.subheader("Equipment Energy Breakdown (Traceable Calculations)")
            breakdown_df = pd.DataFrame(baseline_calc["per_equipment_energy"])
            st.dataframe(
                breakdown_df[[
                    "equipment_id",
                    "equipment_name",
                    "equipment_type",
                    "rated_power_kw",
                    "quantity",
                    "hours_per_day",
                    "operating_days",
                    "utilization_factor",
                    "estimated_energy_kwh",
                    "calculation_trace",
                ]],
                use_container_width=True,
            )

            with st.expander("📌 Baseline Model Assumptions & Traceability", expanded=False):
                for k, v in baseline_calc["assumptions"].items():
                    st.markdown(f"- **{k}**: {v}")

        elif mode_choice == "Demo Scenario":
            st.info("🏷️ **Demo dataset — simulated**: Synthetic hourly office profile with injected operational waste scenarios.")
            demo_csv = DEMO_DATA_DIR / "demo_office_energy.csv"
            demo_manifest_file = DEMO_DATA_DIR / "demo_manifest.json"

            if demo_csv.exists():
                df = pd.read_csv(demo_csv)
                val_info = {
                    "row_count": len(df),
                    "start": df["timestamp"].iloc[0],
                    "end": df["timestamp"].iloc[-1],
                }

                m1, m2, m3 = st.columns(3)
                m1.metric("Simulated Rows", f"{len(df):,}")
                m2.metric("Date Span", f"{val_info['start'][:10]} to {val_info['end'][:10]}")
                m3.metric("Data Quality Status", "Deterministic Demo (seed=42)")

                if demo_manifest_file.exists():
                    with open(demo_manifest_file) as f:
                        manifest = json.load(f)
                    with st.expander("🔍 Injected Controlled Waste Anomalies", expanded=True):
                        for anom in manifest.get("injected_anomalies", []):
                            st.markdown(
                                f"- **{anom['date']} ({anom['hours']})**: *{anom['type']}* — {anom['description']}"
                            )

                st.subheader("Demo Energy Timeseries Preview")
                st.dataframe(df.head(50), use_container_width=True)

    # Downstream Placeholders
    with st.expander("3. Forecast", expanded=False):
        st.subheader("ML Energy Forecast")
        st.write("ML models predict expected interval loads compared against baseline models.")

    with st.expander("4. Waste", expanded=False):
        st.subheader("Anomaly & Waste Detection")
        st.write("Deviations above expected operational load identify actionable energy waste.")

    with st.expander("5. Optimize", expanded=False):
        st.subheader("Constrained Energy Optimization")
        st.write("Peak-load shaving and runtime reallocation within operational flexibility boundaries.")

    with st.expander("6. Finance", expanded=False):
        st.subheader("Financial Impact & Modeled Savings")
        st.write("Modeled cost savings in ₹, energy reductions in kWh, and efficiency incentive matches.")

    with st.expander("7. Action Plan", expanded=False):
        st.subheader("Final Energy Action Plan")
        st.write("Executive summary of scheduled actions and ROI impact projections.")


if __name__ == "__main__":
    main()
