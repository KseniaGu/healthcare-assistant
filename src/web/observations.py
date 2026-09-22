"""Prepare stored observations for faithful display and numeric-only charts."""

import math

import pandas as pd

COLUMNS = ["Test", "Result", "Unit", "Reference", "Date", "Flag", "Numeric value", "Qualifier", "Series"]


def observations_dataframe(observations) -> pd.DataFrame:
    rows = []
    for observation in observations:
        raw = observation.observed_value_text
        qualifier = observation.observed_value_inequality or ""
        numeric = observation.observed_value
        # Old rows without source text remain readable after the additive migration.
        if raw is None and numeric is not None:
            raw = f"{qualifier}{numeric:g}"
        unit = observation.unit or ""
        rows.append(
            {
                "Test": observation.test_name,
                "Result": raw,
                "Unit": unit,
                "Reference": observation.reference_range,
                "Date": observation.observation_date.date() if observation.observation_date else None,
                "Flag": str(observation.flag or "unknown").lower(),
                "Numeric value": numeric,
                "Qualifier": qualifier,
                "Series": f"{observation.test_name} [{unit}]" if unit else observation.test_name,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def chartable_observations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Limits and ranges are not exact point measurements; missing dates are not invented."""
    finite = dataframe["Numeric value"].map(lambda value: pd.notna(value) and math.isfinite(value))
    return dataframe[finite & dataframe["Qualifier"].eq("") & dataframe["Date"].notna()].copy()
