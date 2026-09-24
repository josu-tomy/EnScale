"""
EnScale - ML-assisted energy decision platform for commercial and small industrial buildings.

Streamlit application entrypoint.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config.constants import APP_NAME, APP_TAGLINE, APP_VERSION
from config.settings import settings, DEMO_DATA_DIR, REFERENCE_DATA_DIR
from domain.enums import DataMode
from domain.models import BuildingProfile, Equipment
from services.ingestion_service import (
    load_energy_csv,
    get_energy_csv_template,
    load_demo_energy_data,
    IngestionService,
)
from services.baseline_service import calculate_equipment_energy, BaselineService
from ml.predict import EnergyPredictor
from services.anomaly_service import detect_anomalies
from services.optimization_service import (
    optimize_equipment_schedule,
    InfeasibleConstraintError,
    compute_modeled_savings,
)
from services.impact_service import calculate_impact


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

    # Shared pipeline state
    active_df: pd.DataFrame = None
    predicted_series: np.ndarray = None
    equipment_inventory = None

    # Section 1: Setup
    with st.expander("1. Setup", expanded=False):
        st.subheader("Building & Operational Setup")
        st.write("Configured building baseline profiles, location parameters, and utility tariffs.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Default Tariff", f"₹ {settings.default_tariff_inr_per_kwh:.2f} / kWh")
        col2.metric("Grid Emission Factor", f"{settings.default_emission_factor_kg_per_kwh:.2f} kg CO₂ / kWh")
        col3.metric("Platform Architecture", "100% Local (No External Cloud APIs)")

    # Section 2: Data Screen (Ingestion & Baseline)
    with st.expander("2. Data (Ingestion & Baseline)", expanded=True):
        st.subheader(f"Data Workspace — {mode_choice}")

        if mode_choice == "Use My Historical Data":
            st.markdown("Upload interval energy meter data (15-minute, 30-minute, or hourly).")

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
                    active_df = df
                    st.session_state["active_df"] = df

                    st.success("✅ CSV Successfully Validated and Ingested")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Rows", f"{val_info.get('row_count', len(df)):,}")
                    m2.metric("Sampling Frequency", val_info.get("sampling_frequency", "Unknown"))
                    m3.metric("Duplicate Timestamps", str(val_info.get("duplicate_timestamps_count", 0)))
                    m4.metric("Negative Energy", str(val_info.get("negative_energy_count", 0)))

                    st.markdown(
                        f"**Date Range:** `{val_info.get('date_range_start', 'N/A')}` to `{val_info.get('date_range_end', 'N/A')}`"
                    )

                    warnings = val_info.get("warnings", [])
                    if warnings:
                        with st.expander("⚠️ Data Quality Advisories", expanded=False):
                            for w in warnings:
                                st.warning(w)

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
                    },
                    {
                        "equipment_id": "EQ-02",
                        "equipment_type": "Lighting",
                        "equipment_name": "Building Lighting",
                        "rated_power_kw": 12.0,
                        "quantity": 1,
                        "hours_per_day": 10.0,
                        "operating_days": 22,
                        "utilization_factor": 0.9,
                        "minimum_hours": 10.0,
                        "maximum_hours": 10.0,
                        "is_flexible": False,
                    },
                ]
            st.session_state["equipment_inventory"] = equipment_inventory

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
                active_df = df
                st.session_state["active_df"] = df

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

    # Use session state df if available
    if active_df is None and "active_df" in st.session_state:
        active_df = st.session_state["active_df"]

    # Section 3: Forecast
    with st.expander("3. Forecast", expanded=(active_df is not None)):
        st.subheader("ML Energy Forecast")
        st.write("Machine learning regression model predicting expected interval load patterns.")

        if active_df is not None:
            try:
                predictor = EnergyPredictor()
                if predictor.is_loaded():
                    preds = predictor.predict(active_df)
                    predicted_series = preds
                    st.session_state["predicted_energy"] = preds

                    f_col1, f_col2, f_col3 = st.columns(3)
                    f_col1.metric("Mean Predicted Load", f"{float(np.mean(preds)):.2f} kWh")
                    f_col2.metric("Peak Predicted Load", f"{float(np.max(preds)):.2f} kWh")
                    f_col3.metric("Total Forecast Period Energy", f"{float(np.sum(preds)):,.1f} kWh")

                    # Time-series Plot
                    sample_size = min(len(active_df), 168)  # Preview 1 week (168 hours)
                    preview_df = active_df.head(sample_size).copy()
                    preview_df["predicted_kwh"] = preds[:sample_size]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=preview_df["timestamp"],
                        y=preview_df["energy_kwh"],
                        mode="lines",
                        name="Actual Energy (kWh)",
                        line=dict(color="#1f77b4", width=2),
                    ))
                    fig.add_trace(go.Scatter(
                        x=preview_df["timestamp"],
                        y=preview_df["predicted_kwh"],
                        mode="lines",
                        name="Predicted Baseline (kWh)",
                        line=dict(color="#ff7f0e", width=2, dash="dash"),
                    ))
                    fig.update_layout(
                        title=f"Actual vs Predicted Load Profile (First {sample_size} Hours)",
                        xaxis_title="Timestamp",
                        yaxis_title="Energy (kWh)",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=360,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("ML model artifact not found. Please ensure `ml/artifacts/model.joblib` is trained.")
            except Exception as e:
                st.error(f"Error computing forecast: {e}")
        else:
            st.info("Please load historical or demo data in Section 2 to view ML forecast.")

    # Section 4: Waste & Anomaly Detection
    with st.expander("4. Waste", expanded=(active_df is not None and predicted_series is not None)):
        st.subheader("Anomaly & Waste Detection")
        st.write("Transparent residual-based detection using rolling residual distribution and robust scale estimation.")

        if active_df is not None and predicted_series is not None:
            anomaly_df = detect_anomalies(
                actual_energy_kwh=active_df["energy_kwh"],
                predicted_energy_kwh=predicted_series,
                timestamp=active_df["timestamp"],
                tariff_inr_per_kwh=settings.default_tariff_inr_per_kwh,
            )
            st.session_state["anomaly_df"] = anomaly_df

            flagged_count = int(anomaly_df["anomaly_flag"].sum())
            total_waste_kwh = float(anomaly_df["waste_kwh"].sum())
            total_waste_cost = float(anomaly_df["potential_waste_cost_inr"].sum())

            w1, w2, w3 = st.columns(3)
            w1.metric("Flagged Anomaly Intervals", f"{flagged_count} intervals")
            w2.metric("Total Actionable Waste", f"{total_waste_kwh:,.1f} kWh")
            w3.metric("Estimated Cost of Waste", f"₹ {total_waste_cost:,.2f}")

            if flagged_count > 0:
                st.markdown("### Detected Energy Waste Events")
                display_cols = [
                    "timestamp",
                    "actual_energy_kwh",
                    "predicted_energy_kwh",
                    "forecast_error_kwh",
                    "forecast_error_pct",
                    "severity",
                    "waste_kwh",
                    "potential_waste_cost_inr",
                ]
                anomalies_only = anomaly_df[anomaly_df["anomaly_flag"] == True][display_cols]
                st.dataframe(anomalies_only, use_container_width=True)
            else:
                st.success("No operational waste anomalies detected within the threshold.")
        else:
            st.info("Run the forecast step first to perform residual-based anomaly detection.")

    # Section 5: Constrained Energy Optimization
    optimization_result = None
    with st.expander("5. Optimize", expanded=True):
        st.subheader("Constrained Energy Optimization")
        st.write(
            "Deterministic schedule optimization allocating flexible equipment runtime "
            "strictly within operational window and duration boundaries."
        )

        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            building_area = st.number_input("Net Built-up Floor Area (m²)", min_value=100.0, value=2500.0, step=100.0)
            tariff_input = st.number_input("Electricity Tariff (₹/kWh)", min_value=1.0, value=float(settings.default_tariff_inr_per_kwh), step=0.25)
        with opt_col2:
            op_start = st.number_input("Facility Opening Hour (0-23)", min_value=0, max_value=23, value=8)
            op_end = st.number_input("Facility Closing Hour (0-23)", min_value=0, max_value=23, value=18)

        # Standard Building Profile
        b_profile = BuildingProfile(
            building_id="BLD-CURRENT",
            building_type="Commercial Facility",
            location_state="Maharashtra",
            climate_zone="Composite",
            net_built_up_area_m2=float(building_area),
            operating_start=int(op_start),
            operating_end=int(op_end),
            operating_days_per_month=22,
            occupancy=120.0,
            monthly_budget_inr=150000.0,
            tariff_inr_per_kwh=float(tariff_input),
        )

        # Configurable Equipment Inventory
        eq_items = [
            Equipment(
                equipment_id="EQ-HVAC-01",
                equipment_type="hvac",
                equipment_name="Chiller Plant (Flexible)",
                rated_power_kw=35.0,
                quantity=1,
                hours_per_day=9.0,
                operating_days=22,
                utilization_factor=0.8,
                minimum_hours=7.0,
                maximum_hours=9.0,
                is_flexible=True,
            ),
            Equipment(
                equipment_id="EQ-PUMP-01",
                equipment_type="pump",
                equipment_name="Water Circulation Pump (Flexible)",
                rated_power_kw=15.0,
                quantity=2,
                hours_per_day=6.0,
                operating_days=22,
                utilization_factor=0.85,
                minimum_hours=4.0,
                maximum_hours=6.0,
                is_flexible=True,
            ),
            Equipment(
                equipment_id="EQ-LIGHT-01",
                equipment_type="lighting",
                equipment_name="Base Lighting (Non-Flexible)",
                rated_power_kw=10.0,
                quantity=1,
                hours_per_day=10.0,
                operating_days=22,
                utilization_factor=0.9,
                minimum_hours=10.0,
                maximum_hours=10.0,
                is_flexible=False,
            ),
        ]

        try:
            optimization_result = optimize_equipment_schedule(
                building_profile=b_profile,
                equipment_list=eq_items,
                tariff_inr_per_kwh=float(tariff_input),
            )
            st.session_state["optimization_result"] = optimization_result

            # Modeled Savings Metrics
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Baseline Energy", f"{optimization_result.baseline_energy_kwh:,.1f} kWh")
            o2.metric("Optimized Energy", f"{optimization_result.optimized_energy_kwh:,.1f} kWh")
            o3.metric("Modeled Energy Savings", f"{optimization_result.energy_savings_kwh:,.1f} kWh ({optimization_result.savings_pct:.1f}%)")
            o4.metric("Modeled Cost Savings", f"₹ {optimization_result.cost_savings_inr:,.2f}")

            # Explicit savings statement adhering strictly to forbidden words rule
            st.info(f"📋 **Savings Statement:** {optimization_result.savings_statement}")

            # Detailed Schedule Recommendations
            st.subheader("Equipment Operating Schedules & Recommendations")
            sched_df = pd.DataFrame(optimization_result.schedule_recommendations)
            st.dataframe(sched_df, use_container_width=True)

        except InfeasibleConstraintError as ice:
            st.error(f"❌ Infeasible Constraints: {ice}")
        except Exception as e:
            st.error(f"Optimization error: {e}")

    # Section 6: Financial, Carbon, and EPI/MEPI Impact
    with st.expander("6. Finance", expanded=(optimization_result is not None)):
        st.subheader("Financial Impact, Carbon Avoided & EPI / MEPI Normalization")
        st.write("Holistic environmental accounting, annual financial scaling, and normalized performance indices.")

        if optimization_result is not None:
            impact = calculate_impact(
                optimization_result=optimization_result,
                net_built_up_area_m2=float(building_area),
                months_per_year=12.0,
            )

            # Impact Metrics
            i1, i2, i3 = st.columns(3)
            i1.metric("Monthly Cost Savings", f"₹ {impact.monthly_cost_savings:,.2f}")
            i1.metric("Annual Cost Savings", f"₹ {impact.annual_cost_savings:,.2f}")

            i2.metric("Monthly Energy Savings", f"{impact.monthly_energy_savings:,.1f} kWh")
            i2.metric("Annual Energy Savings", f"{impact.annual_energy_savings:,.1f} kWh")

            i3.metric("Annual CO₂ Avoided", f"{impact.co2_avoided_kg:,.1f} kg CO₂")
            i3.caption(impact.co2_factor_statement)

            st.divider()

            # EPI / MEPI Benchmarks
            st.subheader("Energy Performance Indices (EPI / MEPI)")
            epi1, epi2 = st.columns(2)
            epi1.metric(
                "Modeled EPI (MEPI)",
                f"{impact.mepi_kwh_m2_year:.2f} kWh/m²/year",
                help="MEPI = Modeled Annual Energy / Net Built-up Area",
            )
            epi2.metric(
                "Measured EPI",
                impact.measured_epi_status,
                help="Measured EPI requires verified 12-month interval meter data",
            )

            st.markdown(
                "> **Area Normalization Rule:** "
                "Electricity consumption is derived strictly from operational equipment and load models, "
                "never inferred from building area alone. Floor area is used solely as a normalizing denominator "
                "for benchmarking efficiency (kWh/m²/year)."
            )

            with st.expander("📌 Annualization & Emissions Assumptions", expanded=False):
                for k, v in impact.annualization_assumptions.items():
                    st.markdown(f"- **{k}**: {v}")
        else:
            st.info("Run the constrained optimization in Section 5 to calculate financial and emissions impacts.")

    # Section 7: Action Plan
    with st.expander("7. Action Plan", expanded=(optimization_result is not None)):
        st.subheader("Final Energy Action Plan")
        st.write("Actionable decision roadmap consolidating schedule adjustments, savings projections, and operational safeguards.")

        if optimization_result is not None:
            action_rows = []
            for item in optimization_result.schedule_recommendations:
                action_rows.append({
                    "Equipment": item["equipment_name"],
                    "Action Type": "Runtime Rescheduling" if item["is_flexible"] else "Maintain Base Operation",
                    "Recommended Hours": f"{item['hours_per_day']:.1f} hrs/day",
                    "Modeled Monthly Energy": f"{item['estimated_energy_kwh']:,.1f} kWh",
                    "Monthly Cost": f"₹ {item['cost_inr']:,.2f}",
                    "Operational Constraint Status": "Within safe operating window",
                })
            action_df = pd.DataFrame(action_rows)
            st.dataframe(action_df, use_container_width=True)

            st.success("✅ Action plan verified: All operational constraints satisfied with zero constraint violations.")
        else:
            st.info("Complete upstream steps to generate the energy action plan.")


if __name__ == "__main__":
    main()
