"""
Setup script to initialize reference and demo datasets for EnScale.

Generates local deterministic demo data adhering strictly to canonical contracts:
- data/demo/demo_office_energy.csv (Labeled: "Demo dataset — simulated")
- data/demo/demo_equipment.json
- data/demo/demo_manifest.json
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

    # Label: "Demo dataset — simulated"
    # Seed: 42 (deterministic)
    np.random.seed(42)

    # Generate 62 days (July 1 to August 31, 2026) of hourly energy data
    date_range = pd.date_range(start="2026-07-01 00:00:00", end="2026-08-31 23:00:00", freq="h")
    records = []

    for dt in date_range:
        hour = dt.hour
        is_sunday = dt.dayofweek == 6
        is_saturday = dt.dayofweek == 5
        day_idx = (dt - date_range[0]).days
        day_of_week = dt.dayofweek

        # Multi-day weather cycle (warmer vs cooler days)
        weather_variation = 3.2 * np.sin(day_idx * 2 * np.pi / 8.5)
        temp_c = 26.5 + weather_variation + 7.0 * np.sin((hour - 9) * np.pi / 12) + np.random.normal(0, 0.6)

        # Baseline occupancy and base energy
        if is_sunday or hour < 7 or hour >= 19:
            occupancy = float(max(0, np.random.poisson(2)))
            base_load = 17.5 + np.random.normal(0, 1.0)
        elif is_saturday:
            # Saturday partial half-day activity (09:00 - 14:00)
            if 9 <= hour <= 14:
                occupancy = float(np.random.normal(45, 5))
                cooling = max(0.0, (temp_c - 24.0) * 2.2)
                base_load = 48.0 + cooling + (occupancy * 0.12) + np.random.normal(0, 2.0)
            else:
                occupancy = float(max(0, np.random.poisson(3)))
                base_load = 20.0 + np.random.normal(0, 1.2)
        else:
            # Weekday working hours: 08:00 - 18:00 (with realistic day-to-day variance)
            weekday_occupancy_scale = [120, 148, 155, 142, 126][day_of_week]
            occupancy = float(np.random.normal(weekday_occupancy_scale, 7))
            cooling_demand = max(0.0, (temp_c - 23.5) * 4.0)
            # Midday lunch dip (12:00 - 13:00) and peak afternoon (14:00 - 16:00)
            hour_factor = 1.12 if 13 <= hour <= 16 else (0.92 if hour in [8, 12, 18] else 1.0)
            base_load = (78.0 + cooling_demand + (occupancy * 0.16)) * hour_factor + np.random.normal(0, 2.5)

        # Controlled anomaly injections
        date_str = dt.strftime("%Y-%m-%d")
        is_anomaly = False
        anomaly_type = None

        # Anomaly 1: Friday night HVAC left running past closing (2026-07-03, 19:00 - 23:00)
        if date_str == "2026-07-03" and 19 <= hour <= 23:
            base_load += 68.0  # Cooling chillers left active overnight
            is_anomaly = True
            anomaly_type = "HVAC operating after closing"

        # Anomaly 2: Sunday lighting left active overnight (2026-07-05, 01:00 - 06:00)
        elif date_str == "2026-07-05" and 1 <= hour <= 6:
            base_load += 24.0  # Floor lighting left on
            is_anomaly = True
            anomaly_type = "Lighting operating overnight"

        # Anomaly 3: HVAC operating after closing in August (2026-08-10, 19:00 - 23:00)
        elif date_str == "2026-08-10" and 19 <= hour <= 23:
            base_load += 65.0  # HVAC chillers left running
            is_anomaly = True
            anomaly_type = "HVAC operating after closing"

        # Anomaly 4: Lighting operating overnight in August (2026-08-16, 01:00 - 06:00)
        elif date_str == "2026-08-16" and 1 <= hour <= 6:
            base_load += 22.0  # Lighting floors left on overnight
            is_anomaly = True
            anomaly_type = "Lighting operating overnight"

        # Anomaly 5: Compressor failure to unload (2026-08-24, 12:00 - 17:00)
        elif date_str == "2026-08-24" and 12 <= hour <= 17:
            base_load += 38.0  # Compressor failure to unload
            is_anomaly = True
            anomaly_type = "Compressor extended operation"

        energy_kwh = max(3.0, base_load)
        eq_load_kw = energy_kwh * 0.94

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
    print(f"Generated demo energy data: {csv_path} ({len(energy_df)} rows) — Label: 'Demo dataset — simulated'")

    # Manifest documenting demo simulation
    manifest = {
        "dataset_name": "Demo Commercial Office Energy Profile",
        "dataset_label": "Demo dataset — simulated",
        "source": "Synthetically generated with deterministic seed (seed=42)",
        "is_real_world_measurement": False,
        "rows_count": len(energy_df),
        "date_range": {
            "start": "2026-07-01 00:00:00",
            "end": "2026-08-31 23:00:00",
        },
        "sampling_frequency": "1 hour",
        "injected_anomalies": [
            {
                "date": "2026-08-10",
                "hours": "19:00 - 23:00",
                "type": "HVAC operating after closing",
                "description": "Chiller and AHU units remained at day setpoint after office closing.",
            },
            {
                "date": "2026-08-16",
                "hours": "01:00 - 06:00",
                "type": "Lighting operating overnight",
                "description": "Floor LED arrays left energized during empty Sunday overnight.",
            },
            {
                "date": "2026-08-24",
                "hours": "12:00 - 17:00",
                "type": "Compressor extended operation",
                "description": "Air compressor failed to unload, running continuous full-duty cycle.",
            },
        ],
    }

    manifest_path = DEMO_DATA_DIR / "demo_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated demo manifest: {manifest_path}")

    # Demo equipment inventory
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
