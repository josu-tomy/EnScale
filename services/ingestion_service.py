"""
Data ingestion service for EnScale.

Supports the three data modes:
- MODE 1: User historical data
- MODE 2: Equipment-based baseline
- MODE 3: Demo scenario
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import pandas as pd

from domain.enums import DataMode
from domain.models import BuildingProfile, Equipment
from domain.schemas import ValidationResult
from services.validation_service import (
    validate_energy_dataframe,
    validate_building_profile,
    validate_equipment,
)


class IngestionService:
    """Service interface for ingesting building and energy data across all three modes."""

    def __init__(self, data_mode: DataMode = DataMode.MODE_3_DEMO):
        self.data_mode = data_mode

    def ingest_historical_csv(self, filepath_or_buffer: Any) -> pd.DataFrame:
        """
        MODE 1: Ingests user historical energy consumption CSV.
        Required columns: timestamp, energy_kwh.
        Optional columns: temperature_c, occupancy, equipment_load_kw.
        """
        df = pd.read_csv(filepath_or_buffer)
        val_res = validate_energy_dataframe(df)
        if not val_res.is_valid:
            raise ValueError(f"Historical data validation failed: {'; '.join(val_res.errors)}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def ingest_equipment_inventory(self, items: List[Dict[str, Any]]) -> List[Equipment]:
        """
        MODE 2: Ingests equipment inventory for equipment-based baseline calculations.
        """
        equipment_list: List[Equipment] = []
        for item in items:
            eq = Equipment.from_dict(item) if isinstance(item, dict) else item
            val_res = validate_equipment(eq)
            if not val_res.is_valid:
                raise ValueError(f"Equipment validation failed for {eq.equipment_id}: {'; '.join(val_res.errors)}")
            equipment_list.append(eq)
        return equipment_list

    def load_demo_scenario(self) -> Dict[str, Any]:
        """
        MODE 3: Loads reference demo scenario for testing and evaluation.
        """
        demo_building = BuildingProfile(
            building_id="BLD-DEMO-001",
            building_type="Commercial Office",
            location_state="Maharashtra",
            climate_zone="Composite",
            net_built_up_area_m2=2500.0,
            operating_start=9,
            operating_end=18,
            operating_days_per_month=22,
            occupancy=150.0,
            monthly_budget_inr=150000.0,
            tariff_inr_per_kwh=8.50,
        )
        return {
            "mode": DataMode.MODE_3_DEMO,
            "building_profile": demo_building,
            "description": "Standard 2500 m2 commercial office demo scenario in Mumbai (Composite climate).",
        }
