"""Synthetic data generators.

These emit dataframes with the *exact raw schema* of the real Kaggle datasets,
but with deliberately planted signal so the trained models score meaningfully
above chance. That lets the entire pipeline (preprocess -> train -> export ->
serve -> load test) run end-to-end with no Kaggle account and no multi-GB
download. The feature-engineering and training code is identical whether the
rows come from here or from the real CSVs.

Real metrics (AUC > 0.76, RMSE < $3.50) require the real data; see the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(42)


def _hashlike(prefix: str, n_categories: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Avazu-style 8-char hex-ish category labels."""
    return np.array([f"{prefix}{i:06x}" for i in range(n_categories)])


def make_avazu(n_rows: int, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Synthetic Avazu CTR rows, time-ordered across 10 days (days 21-30)."""
    rng = rng or _RNG

    # Downscaled cardinalities (real Avazu is far larger) — keeps the demo fast
    # while still exercising the frequency/label encoders.
    site_ids = _hashlike("s", 200, rng)
    site_domains = _hashlike("sd", 150, rng)
    app_ids = _hashlike("a", 120, rng)
    app_domains = _hashlike("ad", 60, rng)
    device_ids = _hashlike("d", 5000, rng)
    device_models = _hashlike("dm", 300, rng)

    # Power-law popularity so frequency encoding carries signal.
    def popular(choices: np.ndarray) -> np.ndarray:
        w = 1.0 / (1 + np.arange(len(choices)))
        w /= w.sum()
        return rng.choice(choices, size=n_rows, p=w)

    # Time: 10 ordered days, hours 0-23. Format YYMMDDHH (year=14, month=10).
    frac = np.sort(rng.random(n_rows))           # ordered -> time-ordered rows
    day = (21 + (frac * 10).astype(int)).clip(21, 30)
    hour_of_day = rng.integers(0, 24, n_rows)
    hour = np.array([f"14{10:02d}{d:02d}{h:02d}" for d, h in zip(day, hour_of_day)])

    banner_pos = rng.integers(0, 8, n_rows)
    device_type = rng.integers(0, 6, n_rows)
    device_conn_type = rng.integers(0, 6, n_rows)
    site_id = popular(site_ids)

    df = pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "hour": hour,
            "C1": rng.integers(1000, 1012, n_rows),
            "banner_pos": banner_pos,
            "site_id": site_id,
            "site_domain": popular(site_domains),
            "site_category": _hashlike("sc", 26, rng)[rng.integers(0, 26, n_rows)],
            "app_id": popular(app_ids),
            "app_domain": popular(app_domains),
            "app_category": _hashlike("ac", 36, rng)[rng.integers(0, 36, n_rows)],
            "device_id": rng.choice(device_ids, n_rows),
            "device_ip": _hashlike("ip", n_rows, rng),  # near-unique, will be dropped
            "device_model": popular(device_models),
            "device_type": device_type,
            "device_conn_type": device_conn_type,
        }
    )
    for i, c in enumerate(["C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]):
        df[c] = rng.integers(0, 50 + i * 30, n_rows)

    # --- Plant signal: click probability is a logistic function of a few drivers.
    site_rank = pd.Series(site_id).map(
        {s: r for r, s in enumerate(site_ids)}
    ).to_numpy()
    site_popularity = 1.0 / (1 + site_rank)                  # popular sites click more
    rush = ((hour_of_day >= 17) & (hour_of_day <= 22)).astype(float)
    logit = (
        -1.6
        + 1.8 * site_popularity
        + 0.5 * rush
        + 0.15 * (banner_pos == 0)
        - 0.1 * device_type
        + rng.normal(0, 0.5, n_rows)
    )
    p = 1 / (1 + np.exp(-logit))
    df["click"] = (rng.random(n_rows) < p).astype(int)

    # Real Avazu column order has click second; not required, but tidy.
    cols = ["id", "click"] + [c for c in df.columns if c not in ("id", "click")]
    return df[cols]


def make_nyc_taxi(n_rows: int, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Synthetic NYC Taxi rows, time-ordered across 2009-2015."""
    rng = rng or _RNG

    frac = np.sort(rng.random(n_rows))
    years = (2009 + (frac * 6).astype(int)).clip(2009, 2015)
    months = rng.integers(1, 13, n_rows)
    days = rng.integers(1, 28, n_rows)
    hours = rng.integers(0, 24, n_rows)
    pickup_dt = pd.to_datetime(
        {"year": years, "month": months, "day": days, "hour": hours}
    )

    # NYC bounding box.
    plat = rng.uniform(40.65, 40.82, n_rows)
    plon = rng.uniform(-74.02, -73.92, n_rows)
    dlat = plat + rng.normal(0, 0.03, n_rows)
    dlon = plon + rng.normal(0, 0.03, n_rows)
    passengers = rng.integers(1, 7, n_rows)

    # True (latent) distance drives fare.
    R = 6371.0
    phi1, phi2 = np.radians(plat), np.radians(dlat)
    dphi = np.radians(dlat - plat)
    dlam = np.radians(dlon - plon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    trip_km = 2 * R * np.arcsin(np.sqrt(a))

    rush = ((hours >= 7) & (hours <= 9)) | ((hours >= 17) & (hours <= 19))
    fare = (
        2.5
        + 1.9 * trip_km
        + 1.5 * rush
        + 0.05 * (years - 2009)            # mild fare inflation over time
        + rng.normal(0, 1.2, n_rows)
    ).clip(2.5, 500)

    return pd.DataFrame(
        {
            "key": [f"{i}" for i in range(n_rows)],
            "fare_amount": np.round(fare, 2),
            "pickup_datetime": pickup_dt.astype(str),
            "pickup_longitude": plon,
            "pickup_latitude": plat,
            "dropoff_longitude": dlon,
            "dropoff_latitude": dlat,
            "passenger_count": passengers,
        }
    )
