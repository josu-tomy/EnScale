"""
Training script to build and evaluate the forecast model and output artifacts.
"""

import sys
import json
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DEMO_DATA_DIR
from ml.train import train_and_evaluate
from scripts.setup_data import generate_demo_dataset


def run_training() -> None:
    demo_csv = DEMO_DATA_DIR / "demo_office_energy.csv"
    demo_manifest_file = DEMO_DATA_DIR / "demo_manifest.json"

    if not demo_csv.exists() or not demo_manifest_file.exists():
        print("Demo data not found. Generating demo dataset first...")
        generate_demo_dataset()

    print(f"Loading training data from {demo_csv}...")
    df = pd.read_csv(demo_csv)

    manifest = {}
    if demo_manifest_file.exists():
        with open(demo_manifest_file) as f:
            manifest = json.load(f)

    print("Executing ML training pipeline (HistGradientBoostingRegressor vs Historical Baseline)...")
    metadata = train_and_evaluate(
        df=df,
        dataset_name=manifest.get("dataset_name", "Demo Commercial Office Energy Profile"),
        dataset_manifest=manifest,
        target_col="energy_kwh",
        model_type="HistGradientBoostingRegressor",
    )

    print("\n--- Training Completed Successfully ---")
    print(f"Model Type: {metadata['model_type']}")
    print(f"Train Rows: {metadata['train_rows']} | Val Rows: {metadata['validation_rows']} | Test Rows: {metadata['test_rows']}")
    val_comp = metadata['metrics']['validation']
    print(f"Validation Baseline MAE: {val_comp['baseline_metrics']['MAE']:.2f} kWh")
    print(f"Validation ML MAE:       {val_comp['ml_metrics']['MAE']:.2f} kWh")
    print(f"Improvement vs Baseline: {val_comp['improvement_vs_baseline_pct']:.2f}%")
    print(f"Acceptance Rule Passed:  {val_comp['ml_outperformed_baseline']}")

    test_comp = metadata['metrics']['test']
    print(f"\nFinal Test ML MAE:       {test_comp['ml_metrics']['MAE']:.2f} kWh")
    print(f"Final Test ML R²:        {test_comp['ml_metrics']['R2']:.4f}")
    print(f"Final Test ML CV(RMSE):  {test_comp['ml_metrics']['CV(RMSE)']:.2f}%")


if __name__ == "__main__":
    run_training()
