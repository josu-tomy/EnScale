"""
Training script to build the initial baseline forecast model from demo data.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DEMO_DATA_DIR
from ml.train import train_baseline_model
from scripts.setup_data import generate_demo_dataset


def run_training() -> None:
    demo_csv = DEMO_DATA_DIR / "demo_office_energy.csv"
    if not demo_csv.exists():
        print("Demo data not found. Generating demo dataset first...")
        generate_demo_dataset()

    print(f"Loading training data from {demo_csv}...")
    df = pd.read_csv(demo_csv)
    artifact_path = train_baseline_model(df, model_name="energy_forecast_model.joblib", model_type="ridge")
    print(f"Model trained and saved successfully to {artifact_path}")


if __name__ == "__main__":
    run_training()
