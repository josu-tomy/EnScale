"""
Tests for data ingestion, CSV validation, and data mode contracts.
"""

import io
import pandas as pd
import pytest

from domain.enums import DataMode
from domain.models import Equipment
from services.ingestion_service import (
    load_energy_csv,
    get_energy_csv_template,
    IngestionService,
)


def test_load_energy_csv_valid():
    csv_data = io.StringIO(
        "timestamp,energy_kwh,temperature_c\n"
        "2026-08-01 01:00:00,42.1,27.8\n"
        "2026-08-01 00:00:00,45.2,28.5\n"
        "2026-08-01 02:00:00,40.0,27.1\n"
    )
    df = load_energy_csv(csv_data)
    assert len(df) == 3
    assert "validation_info" in df.attrs
    val_info = df.attrs["validation_info"]
    assert val_info["is_valid"] is True
    assert val_info["row_count"] == 3
    assert val_info["sampling_frequency"] == "Hourly"
    # Verify chronological sorting was performed
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-08-01 00:00:00")
    assert df["timestamp"].iloc[-1] == pd.Timestamp("2026-08-01 02:00:00")


def test_load_energy_csv_missing_required_column():
    csv_data = io.StringIO(
        "timestamp,power_kw\n"
        "2026-08-01 00:00:00,45.2\n"
    )
    with pytest.raises(ValueError, match="Missing required energy column"):
        load_energy_csv(csv_data)


def test_load_energy_csv_prohibited_alias():
    # Prohibited alias 'energy' instead of 'energy_kwh'
    csv_data = io.StringIO(
        "timestamp,energy\n"
        "2026-08-01 00:00:00,45.2\n"
    )
    with pytest.raises(ValueError, match="Prohibited alias column 'energy'"):
        load_energy_csv(csv_data)


def test_load_energy_csv_duplicate_timestamps():
    csv_data = io.StringIO(
        "timestamp,energy_kwh\n"
        "2026-08-01 00:00:00,45.2\n"
        "2026-08-01 00:00:00,46.0\n"
        "2026-08-01 01:00:00,42.1\n"
    )
    df = load_energy_csv(csv_data)
    # Does not silently discard duplicates
    assert len(df) == 3
    val_info = df.attrs["validation_info"]
    assert val_info["duplicate_timestamps_count"] == 1
    assert any("duplicate timestamp" in w for w in val_info["warnings"])


def test_load_energy_csv_negative_energy():
    csv_data = io.StringIO(
        "timestamp,energy_kwh\n"
        "2026-08-01 00:00:00,45.2\n"
        "2026-08-01 01:00:00,-10.0\n"
    )
    with pytest.raises(ValueError, match="negative energy consumption"):
        load_energy_csv(csv_data)


def test_csv_template_download():
    template = get_energy_csv_template()
    assert "timestamp,energy_kwh,temperature_c,occupancy,equipment_load_kw" in template
    lines = template.strip().split("\n")
    assert len(lines) >= 2


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


def test_load_energy_csv_missing_timestamp():
    """Verifies that missing timestamp column raises a clear validation ValueError."""
    csv_data = io.StringIO("energy_kwh,temperature_c\n45.2,28.5\n")
    with pytest.raises(ValueError, match="Missing required energy column: 'timestamp'"):
        load_energy_csv(csv_data)


def test_load_energy_csv_missing_energy_kwh():
    """Verifies that missing energy_kwh column raises a clear validation ValueError."""
    csv_data = io.StringIO("timestamp,temperature_c\n2026-08-01 00:00:00,28.5\n")
    with pytest.raises(ValueError, match="Missing required energy column: 'energy_kwh'"):
        load_energy_csv(csv_data)


def test_load_energy_csv_invalid_timestamp():
    """Verifies that unparseable timestamp format raises a validation ValueError."""
    csv_data = io.StringIO(
        "timestamp,energy_kwh\n"
        "not-a-valid-date,45.2\n"
        "2026-08-01 01:00:00,42.0\n"
    )
    with pytest.raises(ValueError, match="could not be parsed as datetime format"):
        load_energy_csv(csv_data)


def test_load_energy_csv_empty_file():
    """Verifies that empty file raises a ValueError or EmptyDataError."""
    csv_data = io.StringIO("")
    with pytest.raises(Exception):
        load_energy_csv(csv_data)


def test_load_energy_csv_malformed_values():
    """Verifies that non-numeric energy_kwh values raise a validation ValueError."""
    csv_data = io.StringIO(
        "timestamp,energy_kwh\n"
        "2026-08-01 00:00:00,corrupted_value\n"
    )
    with pytest.raises(ValueError, match="contains .* non-numeric values"):
        load_energy_csv(csv_data)
