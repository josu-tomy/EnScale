"""
Tests for data ingestion and data mode contracts.
"""

import io
import pandas as pd
import pytest

from domain.enums import DataMode
from domain.models import Equipment
from services.ingestion_service import IngestionService


def test_ingest_valid_historical_csv():
    service = IngestionService(DataMode.MODE_1_HISTORICAL)
    csv_data = io.StringIO(
        "timestamp,energy_kwh,temperature_c\n"
        "2026-08-01 00:00:00,45.2,28.5\n"
        "2026-08-01 01:00:00,42.1,27.8\n"
    )
    df = service.ingest_historical_csv(csv_data)
    assert len(df) == 2
    assert "timestamp" in df.columns
    assert "energy_kwh" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_ingest_missing_required_column_raises():
    service = IngestionService(DataMode.MODE_1_HISTORICAL)
    csv_data = io.StringIO(
        "timestamp,power_kw\n"
        "2026-08-01 00:00:00,45.2\n"
    )
    with pytest.raises(ValueError, match="Historical data validation failed"):
        service.ingest_historical_csv(csv_data)


def test_ingest_equipment_inventory():
    service = IngestionService(DataMode.MODE_2_EQUIPMENT_BASELINE)
    items = [
        {
            "equipment_id": "EQ-01",
            "equipment_type": "HVAC",
            "equipment_name": "Main Chiller",
            "rated_power_kw": 30.0,
            "quantity": 1,
            "hours_per_day": 8.0,
            "operating_days": 20,
            "utilization_factor": 0.8,
            "minimum_hours": 4.0,
            "maximum_hours": 10.0,
            "is_flexible": True,
        }
    ]
    eq_list = service.ingest_equipment_inventory(items)
    assert len(eq_list) == 1
    assert eq_list[0].equipment_id == "EQ-01"
    assert eq_list[0].rated_power_kw == 30.0


def test_load_demo_scenario():
    service = IngestionService(DataMode.MODE_3_DEMO)
    demo = service.load_demo_scenario()
    assert demo["mode"] == DataMode.MODE_3_DEMO
    assert demo["building_profile"].building_id == "BLD-DEMO-001"
    assert demo["building_profile"].net_built_up_area_m2 == 2500.0
