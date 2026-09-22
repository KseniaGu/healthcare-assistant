"""Lossless table rendering and conservative chart selection, without Streamlit startup."""

import unittest
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from backend.utils.helpers import observation_values
from configs.data import DataSettings
from src.web.observations import chartable_observations, observations_dataframe


class V4DisplayTests(unittest.TestCase):
    def test_all_value_kinds_remain_visible_but_only_exact_numbers_are_charted(self):
        observations = []
        for value in ("4.50", "0-1", "negative", "< 3.0", "2,30", None, "≈ 5", "+"):
            values = observation_values({"value": value, "unit": "mg/L"}, DataSettings().common_inequalities)
            observations.append(SimpleNamespace(test_name="Synthetic", observation_date=datetime(2025, 1, 1), **values))
        frame = observations_dataframe(observations)
        self.assertEqual(frame["Result"].dropna().tolist(), ["4.50", "0-1", "negative", "< 3.0", "2,30", "≈ 5", "+"])
        self.assertTrue(pd.isna(frame.iloc[5]["Result"]))
        self.assertEqual(chartable_observations(frame)["Numeric value"].tolist(), [4.5, 2.3])
        self.assertEqual(frame["Flag"].tolist(), ["unknown"] * 8)
        self.assertEqual(int(frame["Flag"].isin(["high", "low"]).sum()), 0)

    def test_legacy_values_unknown_dates_and_different_units(self):
        rows = [
            SimpleNamespace(
                test_name="Synthetic",
                observed_value=4.5,
                observed_value_text=None,
                observed_value_inequality="",
                unit=unit,
                reference_range=None,
                flag=None,
                observation_date=date,
            )
            for unit, date in (("mg/L", datetime(2025, 1, 1)), ("g/L", datetime(2025, 1, 2)), (None, None))
        ]
        frame = observations_dataframe(rows)
        self.assertEqual(frame["Result"].tolist(), ["4.5"] * 3)
        self.assertIsNone(frame.iloc[2]["Date"])
        self.assertEqual(chartable_observations(frame)["Series"].tolist(), ["Synthetic [mg/L]", "Synthetic [g/L]"])
        frame["Date"] = None
        self.assertTrue(chartable_observations(frame).empty)
        self.assertTrue(chartable_observations(observations_dataframe([])).empty)


if __name__ == "__main__":
    unittest.main()
