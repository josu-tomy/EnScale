"""
Evaluation script for anomaly detection performance on a separate ground-truth dataset.

Generates a separate test dataset:
- Exactly 80% normal points
- Exactly 20% controlled anomalies:
  * after-hours HVAC
  * overnight lighting
  * extended compressor runtime
  * unusual load spike

Evaluates:
- Precision
- Recall (True Positive Rate)
- F1-Score
- False Positive Rate (FPR)
- True Positive Rate (TPR)

Saves actual test results to ml/artifacts/anomaly_evaluation.json.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import REFERENCE_DATA_DIR, ML_ARTIFACTS_DIR
from services.anomaly_service import detect_anomalies


def create_anomaly_test_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Creates a deterministic synthetic test dataset with exactly 80% normal
    and 20% controlled anomalies.
    """
    np.random.seed(101)  # Separate fixed seed

    n_anomalies = int(n_samples * 0.20)  # exactly 20% = 100
    n_normal = n_samples - n_anomalies   # exactly 80% = 400

    timestamps = pd.date_range("2026-09-01 00:00:00", periods=n_samples, freq="h")

    # Generate baseline/predicted pattern (typical diurnal office cycle)
    hours = timestamps.hour.to_numpy()
    predicted = 50.0 + 35.0 * np.clip(np.sin((hours - 8) * np.pi / 12), 0.0, None)

    # Actual energy starts as predicted + small realistic noise
    noise = np.random.normal(0, 2.0, n_samples)
    actual = np.maximum(5.0, predicted + noise)

    ground_truth = np.zeros(n_samples, dtype=bool)
    anomaly_types = ["none"] * n_samples

    # Inject 100 anomaly hours across realistic operational event episodes (each 3 to 5 hours)
    anomaly_episodes = [
        # after-hours HVAC episodes (4 hours each)
        ("after-hours HVAC", range(20, 24), 45.0),
        ("after-hours HVAC", range(44, 48), 50.0),
        ("after-hours HVAC", range(68, 72), 48.0),
        ("after-hours HVAC", range(92, 96), 46.0),
        ("after-hours HVAC", range(116, 120), 52.0),
        ("after-hours HVAC", range(140, 145), 45.0), # 25 hrs total

        # overnight lighting episodes (5 hours each)
        ("overnight lighting", range(170, 175), 25.0),
        ("overnight lighting", range(194, 199), 24.0),
        ("overnight lighting", range(218, 223), 26.0),
        ("overnight lighting", range(242, 247), 25.0),
        ("overnight lighting", range(266, 271), 25.0), # 25 hrs total

        # extended compressor runtime episodes (5 hours each)
        ("extended compressor runtime", range(295, 300), 38.0),
        ("extended compressor runtime", range(319, 324), 40.0),
        ("extended compressor runtime", range(343, 348), 36.0),
        ("extended compressor runtime", range(367, 372), 42.0),
        ("extended compressor runtime", range(391, 396), 38.0), # 25 hrs total

        # unusual load spikes (3 to 4 hours each)
        ("unusual load spike", range(415, 419), 65.0),
        ("unusual load spike", range(435, 439), 70.0),
        ("unusual load spike", range(455, 459), 60.0),
        ("unusual load spike", range(470, 474), 75.0),
        ("unusual load spike", range(485, 490), 68.0),
        ("unusual load spike", range(495, 499), 65.0), # 25 hrs total
    ]

    for label, idx_range, delta_kwh in anomaly_episodes:
        for idx in idx_range:
            if idx < n_samples:
                actual[idx] += delta_kwh + np.random.normal(0, 1.2)
                ground_truth[idx] = True
                anomaly_types[idx] = label

    df = pd.DataFrame({
        "timestamp": timestamps,
        "predicted_energy_kwh": np.round(predicted, 2),
        "actual_energy_kwh": np.round(actual, 2),
        "ground_truth_anomaly": ground_truth,
        "anomaly_type": anomaly_types,
    })

    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REFERENCE_DATA_DIR / "anomaly_evaluation_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created anomaly test dataset: {csv_path} ({n_normal} normal, {n_anomalies} anomalous)")

    return df


def evaluate_anomaly_detector() -> Dict[str, Any]:
    dataset = create_anomaly_test_dataset(n_samples=500)

    # Run anomaly detection
    results = detect_anomalies(
        actual_energy_kwh=dataset["actual_energy_kwh"],
        predicted_energy_kwh=dataset["predicted_energy_kwh"],
        timestamp=dataset["timestamp"],
        window_size=24,
        threshold_sigma=2.5,
        min_waste_kwh=10.0,
    )

    y_true = dataset["ground_truth_anomaly"].values
    y_pred = results["anomaly_flag"].values

    tp = int(np.sum((y_true == True) & (y_pred == True)))
    fp = int(np.sum((y_true == False) & (y_pred == True)))
    fn = int(np.sum((y_true == True) & (y_pred == False)))
    tn = int(np.sum((y_true == False) & (y_pred == False)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = recall  # TPR is identical to recall

    evaluation_report = {
        "dataset_name": "Controlled Anomaly Evaluation Dataset",
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_samples": len(dataset),
        "normal_samples": int(np.sum(y_true == False)),
        "anomaly_samples": int(np.sum(y_true == True)),
        "normal_ratio": round(float(np.mean(y_true == False)), 2),
        "anomaly_ratio": round(float(np.mean(y_true == True)), 2),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "F1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "true_positive_rate": round(tpr, 4),
        },
        "anomaly_categories_tested": [
            "after-hours HVAC",
            "overnight lighting",
            "extended compressor runtime",
            "unusual load spike",
        ],
    }

    ML_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ML_ARTIFACTS_DIR / "anomaly_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"\n--- Anomaly Detection Evaluation Complete ---")
    print(f"Artifact Saved: {out_json}")
    print(f"Precision:           {evaluation_report['metrics']['precision']:.4f}")
    print(f"Recall (TPR):        {evaluation_report['metrics']['recall']:.4f}")
    print(f"F1-Score:            {evaluation_report['metrics']['F1']:.4f}")
    print(f"False Positive Rate: {evaluation_report['metrics']['false_positive_rate']:.4f}")
    print(f"True Positive Rate:  {evaluation_report['metrics']['true_positive_rate']:.4f}")

    return evaluation_report


if __name__ == "__main__":
    evaluate_anomaly_detector()
