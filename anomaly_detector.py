from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


SENSOR_COLS = [
    "engine_rpm",
    "engine_temp_c",
    "battery_voltage",
    "oil_pressure_psi",
    "fuel_efficiency_mpg",
    "brake_pad_mm",
    "tire_pressure_psi",
]

THRESHOLDS = {
    "engine_temp_c": (70, 105, "Engine Temperature"),
    "battery_voltage": (11.8, 14.4, "Battery Voltage"),
    "oil_pressure_psi": (20, 70, "Oil Pressure"),
    "brake_pad_mm": (0, 4.0, "Brake Pad Thickness"),
}


def detect_anomalies(df: pd.DataFrame, contamination=0.05) -> dict:
    recent = df.tail(7 * 24 * 6)
    X = recent[SENSOR_COLS].fillna(method="ffill")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X_scaled)

    anomaly_count = (labels == -1).sum()
    alerts = []

    for col, (low, high, name) in THRESHOLDS.items():
        latest = recent[col].iloc[-1]
        if latest < low or latest > high:
            sev = "CRITICAL" if (latest < low * 0.9 or latest > high * 1.1) else "WARNING"
            alerts.append(
                {
                    "sensor": name,
                    "current": round(latest, 2),
                    "threshold": f"{low}-{high}",
                    "severity": sev,
                }
            )

    return {
        "anomaly_count": int(anomaly_count),
        "anomaly_rate_pct": round((anomaly_count / len(recent)) * 100, 1),
        "alerts": alerts,
        "latest_readings": {c: round(recent[c].iloc[-1], 2) for c in SENSOR_COLS},
    }