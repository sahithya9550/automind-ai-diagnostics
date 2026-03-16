import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_vehicle_telemetry(
    vehicle_id: str,
    days: int = 30,
    inject_fault: bool = True
) -> pd.DataFrame:
    """Simulate realistic OBD-II vehicle sensor data with optional fault injection."""
    np.random.seed(42)
    n_points = days * 24 * 6  # Every 10 minutes

    timestamps = [
        datetime.now() - timedelta(minutes=i * 10)
        for i in range(n_points)
    ]

    data = {
        "timestamp": timestamps,
        "vehicle_id": vehicle_id,
        "engine_rpm": np.random.normal(2200, 300, n_points).clip(700, 6000),
        "engine_temp_c": np.random.normal(92, 3, n_points).clip(70, 130),
        "battery_voltage": np.random.normal(12.6, 0.2, n_points).clip(11.0, 14.8),
        "oil_pressure_psi": np.random.normal(45, 5, n_points).clip(15, 80),
        "fuel_efficiency_mpg": np.random.normal(32, 2, n_points).clip(18, 55),
        "brake_pad_mm": np.linspace(8.0, 3.5, n_points),
        "tire_pressure_psi": np.random.normal(34, 0.5, n_points).clip(28, 40),
    }

    if inject_fault:
        # Inject progressive engine overheating in last 7 days
        fault_start = n_points - (7 * 24 * 6)
        data["engine_temp_c"][fault_start:] += np.linspace(0, 35, n_points - fault_start)

        # Inject battery degradation
        data["battery_voltage"][fault_start:] -= np.linspace(0, 2.1, n_points - fault_start)

    return pd.DataFrame(data)