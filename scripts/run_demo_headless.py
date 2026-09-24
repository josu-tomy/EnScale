import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import pandas as pd
import numpy as np
from pathlib import Path

from domain.models import BuildingProfile, Equipment
from services.ingestion_service import IngestionService
from services.baseline_service import calculate_equipment_energy, calculate_equipment_coverage
from ml.predict import EnergyPredictor
from services.anomaly_service import detect_anomalies, calculate_injected_anomaly_recall
from services.optimization_service import optimize_equipment_schedule
from services.impact_service import calculate_impact
from services.incentive_service import find_incentives
from services.opportunity_service import identify_energy_opportunities, explain_anomaly_episodes, generate_decision_summary

# 1. Building and Equipment Setup (Demo Scenario)
b = BuildingProfile(
    building_id="BLD-COMM-MUMBAI-01",
    building_type="Commercial Office",
    location_state="Maharashtra",
    climate_zone="Composite",
    net_built_up_area_m2=2500.0,
    operating_start=8,
    operating_end=18,
    operating_days_per_month=22,
    occupancy=140.0,
    monthly_budget_inr=150000.0,
    tariff_inr_per_kwh=8.50,
)

equipment_list = [
    Equipment(
        equipment_id="EQ-HVAC-01",
        equipment_type="HVAC",
        equipment_name="Central Chiller & Air Conditioning",
        rated_power_kw=35.0,
        quantity=1,
        hours_per_day=9.0,
        operating_days=22,
        utilization_factor=0.80,
        minimum_hours=7.0,
        maximum_hours=9.0,
        is_flexible=True,
        needed_from_opening=True,
        needed_until_closing=False,
    ),
    Equipment(
        equipment_id="EQ-PUMP-01",
        equipment_type="Motors & Pumps",
        equipment_name="Water Circulation Pumps",
        rated_power_kw=15.0,
        quantity=2,
        hours_per_day=6.0,
        operating_days=22,
        utilization_factor=0.85,
        minimum_hours=4.0,
        maximum_hours=6.0,
        is_flexible=True,
        needed_until_closing=False,
    ),
    Equipment(
        equipment_id="EQ-LIGHT-01",
        equipment_type="Lighting",
        equipment_name="Workstation & Floor Lighting",
        rated_power_kw=10.0,
        quantity=1,
        hours_per_day=10.0,
        operating_days=22,
        utilization_factor=0.90,
        minimum_hours=10.0,
        maximum_hours=10.0,
        is_flexible=False,
        needed_until_closing=False,
    ),
    Equipment(
        equipment_id="EQ-COMP-01",
        equipment_type="Air Compressors",
        equipment_name="Air Compressors",
        rated_power_kw=18.0,
        quantity=1,
        hours_per_day=8.0,
        operating_days=22,
        utilization_factor=0.80,
        minimum_hours=5.0,
        maximum_hours=8.0,
        is_flexible=True,
        needed_until_closing=False,
    ),
]

from services.ingestion_service import load_energy_csv
df = load_energy_csv("data/demo/demo_office_energy.csv")

# 3. Forecast
predictor = EnergyPredictor()
preds = predictor.predict(df)

actual_kwh = float(df["energy_kwh"].sum())
expected_kwh = float(np.sum(preds))
total_residual = actual_kwh - expected_kwh

# 4. Anomaly Detection
anomaly_df = detect_anomalies(
    actual_energy_kwh=df["energy_kwh"],
    predicted_energy_kwh=preds,
    timestamp=df["timestamp"],
    tariff_inr_per_kwh=b.tariff_inr_per_kwh,
)

explained_episodes = explain_anomaly_episodes(
    anomaly_df=anomaly_df,
    building_profile=b,
    tariff_inr_per_kwh=b.tariff_inr_per_kwh,
    limit=5,
    equipment_list=equipment_list,
)

# 5. Optimization
opt_res = optimize_equipment_schedule(
    building_profile=b,
    equipment_list=equipment_list,
    tariff_inr_per_kwh=b.tariff_inr_per_kwh,
)

# 6. Impact
impact = calculate_impact(
    optimization_result=opt_res,
    net_built_up_area_m2=b.net_built_up_area_m2,
    months_per_year=12.0,
)

# 7. Incentives
matches = find_incentives(
    location_state=b.location_state,
    building_type=b.building_type,
    equipment_list=equipment_list,
)

# 8. Opportunities & Decision Summary
opps = identify_energy_opportunities(
    anomaly_df=anomaly_df,
    building_profile=b,
    equipment_list=equipment_list,
    tariff_inr_per_kwh=b.tariff_inr_per_kwh,
    emission_factor_kg_per_kwh=0.82,
)

dec_summary = generate_decision_summary(opps, opt_res, impact, matches, b, anomaly_df)

# 9. Injected anomaly recall & coverage
recall_info = calculate_injected_anomaly_recall(anomaly_df)
eq_calc = calculate_equipment_energy(equipment_list)
cov_info = calculate_equipment_coverage(eq_calc["total_energy"], df)

print("="*60)
print("1. EXPECTED VS ACTUAL KWH AND TOTAL RESIDUAL")
print("="*60)
print(f"Total Observed Meter Energy: {actual_kwh:,.2f} kWh (over {len(df)} hours / {len(df)/24:.1f} days)")
print(f"Total ML Expected Energy:    {expected_kwh:,.2f} kWh")
print(f"Total Model Residual:        {total_residual:,.2f} kWh ({total_residual/actual_kwh*100:.2f}%)")
print(f"Avoidable Waste Flagged:     {anomaly_df['waste_kwh'].sum():,.2f} kWh across {int(anomaly_df['anomaly_flag'].sum())} hours")

print("\n" + "="*60)
print("2. TOP 3 ANOMALIES")
print("="*60)
for idx, ep in enumerate(explained_episodes[:3], 1):
    print(f"\nAnomaly #{idx}:")
    print(f"  Equipment / Category: {ep['category']}")
    print(f"  Time Period:          {ep['time_period']}")
    print(f"  Expected Hourly:      {ep['expected_avg_kwh']:.2f} kWh/h")
    print(f"  Observed Hourly:      {ep['observed_avg_kwh']:.2f} kWh/h")
    print(f"  Residual / Deviation: +{ep['deviation_avg_kwh']:.2f} kWh/h (Total Waste: {ep['total_waste_kwh']:.2f} kWh, ₹{ep['total_waste_cost_inr']:,.2f})")
    print(f"  Anomaly Score:        {ep['max_z_score']:.2f} sigma ({ep['severity']})")
    print(f"  Likely Condition:     {ep['likely_operational_condition']}")
    print(f"  Confidence Basis:     {ep['confidence']}")

print("\n" + "="*60)
print("3. THE OPPORTUNITY LIST")
print("="*60)
for idx, opp in enumerate(opps, 1):
    print(f"\nOpportunity #{idx}: {opp.opportunity_name} [{opp.opportunity_id}]")
    print(f"  Basis:              {opp.basis} (Headline Verified: {opp.is_headline_verified})")
    print(f"  Category:           {opp.category}")
    print(f"  Binding Constraint: {opp.binding_constraint}")
    print(f"  Current Condition:  {opp.current_condition}")
    print(f"  Expected Condition: {opp.expected_condition}")
    print(f"  Avoidable Energy:   {opp.annual_avoidable_energy_kwh:,.1f} kWh/yr ({opp.estimated_avoidable_energy_kwh:,.1f} kWh/mo)")
    print(f"  Annual Cost Saving: ₹{opp.estimated_annual_cost_savings_inr:,.0f}/yr")
    print(f"  Annual CO2 Avoided: {opp.estimated_annual_co2_impact_kg:,.1f} kg CO2/yr")
    print(f"  Intervention:       {opp.possible_intervention}")
    if opp.source_episodes:
        print(f"  Source Episodes:    {', '.join(opp.source_episodes)}")
    if opp.assumptions_note:
        print(f"  Assumptions Note:   {opp.assumptions_note}")

print("\n" + "="*60)
print("4. CURRENT VS OPTIMIZED SCHEDULE TABLE")
print("="*60)
for r in opt_res.schedule_recommendations:
    print(f"- Equipment: {r['equipment_name']} ({'Flexible' if r['is_flexible'] else 'Non-Flexible'})")
    print(f"  Current Schedule:   {r['current_start_time']} - {r['current_stop_time']} ({r['baseline_hours_per_day']:.1f}h/day, {r['baseline_energy_kwh']:,.1f} kWh/mo)")
    print(f"  Optimized Schedule: {r['optimized_start_time']} - {r['optimized_stop_time']} ({r['optimized_hours_per_day']:.1f}h/day, {r['optimized_energy_kwh']:,.1f} kWh/mo)")
    print(f"  Monthly Savings:    {r['modeled_energy_savings_kwh']:,.1f} kWh/mo, ₹{r['modeled_cost_savings_inr']:,.0f}/mo")
    print(f"  Binding Constraint: {r.get('binding_constraint')}")

print("\n" + "="*60)
print("5. ANNUAL RECONCILED IMPACT & TOTALS")
print("="*60)
print(f"Annual Energy Saved:   {impact.annual_energy_savings:,.1f} kWh / yr")
print(f"Annual Money Saved:    ₹ {impact.annual_cost_savings:,.0f} / yr (₹ {impact.annual_cost_savings/100000:.2f} Lakh/yr)")
print(f"Annual CO2 Avoided:    {impact.co2_avoided_kg:,.1f} kg CO2 / yr ({impact.co2_avoided_kg/1000:.2f} tonnes CO2/yr)")
print(f"Tariff Used:           ₹ {b.tariff_inr_per_kwh:.2f} / kWh")
print(f"Emission Factor Used:  {impact.co2_factor_used} kg CO2/kWh ({impact.co2_factor_statement})")

print("\n" + "="*60)
print("6. POLICY / INCENTIVE MATCHES")
print("="*60)
for idx, m in enumerate(matches, 1):
    inc = m["incentive"]
    p_name = inc.get("name") or inc.get("program_name")
    v_status = inc.get("verification_status", "verified")
    print(f"\nMatch #{idx}: {p_name} ({inc['incentive_id']}) [{v_status.upper()}]")
    print(f"  Target Equipment: {inc.get('technology')}")
    print(f"  Eligibility:      {inc.get('eligibility')}")
    print(f"  Policy Ref:       {inc.get('source') or inc.get('policy_reference')}")
    print(f"  Why Matched:      {' | '.join(m['match_reasons'])}")
    print(f"  Verification:     {m['eligibility_notes']}")

print("\n" + "="*60)
print("7. COVERAGE AND INJECTED RECALL")
print("="*60)
print(f"Equipment Coverage: {cov_info['coverage_pct']}% of metered consumption ({cov_info['modeled_equipment_period_kwh']:,.1f} kWh / {cov_info['total_metered_kwh']:,.1f} kWh)")
print(f"Unmodeled Load:     {cov_info['unmodeled_pct']}% ({cov_info['unmodeled_kwh']:,.1f} kWh)")
print(f"Injected Recall:    {recall_info['detected_injected_episodes']}/{recall_info['total_injected_episodes']} episodes detected ({recall_info['episode_recall_pct']}%)")
print(f"Injected Hours:     {recall_info['detected_injected_hours']}/{recall_info['total_injected_hours']} hours flagged ({recall_info['hourly_recall_pct']}%)")
