"""
Setup script to initialize reference and demo datasets for EnScale.

Generates local demo data adhering strictly to canonical contracts:
- data/demo/demo_office_energy.csv
- data/demo/demo_equipment.json
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd

from config.settings import DEMO_DATA_DIR


def generate_demo_dataset() -> None:
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate 30 days of hourly energy data for a commercial office
    date_range = pd.date_range(start="2026-08-01 00:00:00", end="2026-08-31 23:00:00", freq="h")
    records = []

    np.random.seed(42)
    for dt in date_range:
        hour = dt.hour
        is_weekend = dt.dayofweek >= 5

        # Ambient temperature profile
        temp_c = 26.0 + 8.0 * np.sin((hour - 9) * np.pi / 12) + np.random.normal(0, 1.0)

        # Occupancy profile
        if is_weekend or hour < 8 or hour > 19:
            occupancy = float(np.random.poisson(3))
            base_load = 15.0 + np.random.normal(0, 1.5)
        else:
            occupancy = float(np.random.normal(120, 10))
            base_load = 75.0 + (temp_c - 24.0) * 3.5 + np.random.normal(0, 4.0)

        energy_kwh = max(2.0, base_load)
        eq_load_kw = energy_kwh * 0.95

        records.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "energy_kwh": round(float(energy_kwh), 2),
            "temperature_c": round(float(temp_c), 1),
            "occupancy": round(float(occupancy), 1),
            "equipment_load_kw": round(float(eq_load_kw), 2),
        })

    energy_df = pd.DataFrame(records)
    csv_path = DEMO_DATA_DIR / "demo_office_energy.csv"
    energy_df.to_csv(csv_path, index=False)
    print(f"Generated demo energy data: {csv_path} ({len(energy_df)} rows)")

    # 2. Generate demo equipment inventory
    equipment_data = [
        {
            "equipment_id": "EQ-HVAC-01",
            "equipment_type": "HVAC",
            "equipment_name": "Main Chiller Unit 1",
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
            "equipment_id": "EQ-LTG-02",
            "equipment_type": "Lighting",
            "equipment_name": "Office Floor LED Lighting",
            "rated_power_kw": 12.0,
            "quantity": 1,
            "hours_per_day": 11.0,
            "operating_days": 22,
            "utilization_factor": 0.90,
            "minimum_hours": 8.0,
            "maximum_hours": 12.0,
            "is_flexible": False,
        },
        {
            "equipment_id": "EQ-PUMP-03",
            "equipment_type": "Motors & Pumps",
            "equipment_name": "Chilled Water Circulation Pumps",
            "rated_power_kw": 7.5,
            "quantity": 2,
            "hours_per_day": 10.0,
            "operating_days": 22,
            "utilization_factor": 0.80,
            "minimum_hours": 5.0,
            "maximum_hours": 10.0,
            "is_flexible": True,
        },
    ]

    json_path = DEMO_DATA_DIR / "demo_equipment.json"
    with open(json_path, "w") as f:
        json.dump(equipment_data, f, indent=2)
    print(f"Generated demo equipment inventory: {json_path}")


if __name__ == "__main__":
    generate_demo_dataset()
