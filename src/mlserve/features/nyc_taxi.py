"""NYC Taxi preprocessing: clean bad rows, derive temporal + haversine distance
+ airport-distance features. Produces an 18-feature numeric frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlserve.features.base import BasePreprocessor

# Airport coordinates (lat, lon).
JFK = (40.6413, -73.7781)
LGA = (40.7773, -73.8742)
EWR = (40.6895, -74.1745)

FEATURE_ORDER = [
    "passenger_count", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "hour", "day_of_week", "month", "year", "is_rush", "is_weekend",
    "trip_km",
    "jfk_pickup_km", "jfk_dropoff_km",
    "lga_pickup_km", "lga_dropoff_km",
    "ewr_pickup_km", "ewr_dropoff_km",
]


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class TaxiPreprocessor(BasePreprocessor):
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            (df.fare_amount > 1) & (df.fare_amount < 500)
            & df.pickup_longitude.between(-75, -72)
            & df.pickup_latitude.between(40, 42)
            & df.dropoff_longitude.between(-75, -72)
            & df.dropoff_latitude.between(40, 42)
            & df.passenger_count.between(1, 6)
        ].copy()

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(df["pickup_datetime"])
        df = df.assign(_t=dt).sort_values("_t").reset_index(drop=True)  # time order
        df["hour"] = df["_t"].dt.hour
        df["day_of_week"] = df["_t"].dt.dayofweek
        df["month"] = df["_t"].dt.month
        df["year"] = df["_t"].dt.year
        df["is_rush"] = (
            ((df.hour >= 7) & (df.hour <= 9)) | ((df.hour >= 17) & (df.hour <= 19))
        ).astype(int)
        df["is_weekend"] = (df.day_of_week >= 5).astype(int)
        df["trip_km"] = haversine(
            df.pickup_latitude, df.pickup_longitude,
            df.dropoff_latitude, df.dropoff_longitude,
        )
        for name, (lat, lon) in {"jfk": JFK, "lga": LGA, "ewr": EWR}.items():
            df[f"{name}_pickup_km"] = haversine(df.pickup_latitude, df.pickup_longitude, lat, lon)
            df[f"{name}_dropoff_km"] = haversine(df.dropoff_latitude, df.dropoff_longitude, lat, lon)

        df = df.drop(columns=["pickup_datetime", "_t"], errors="ignore")
        ordered = [c for c in FEATURE_ORDER if c in df.columns]
        return df[ordered + [self.config.target]]
