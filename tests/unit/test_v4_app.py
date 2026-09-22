"""Streamlit smoke test with a fake processor; no model or database initialization."""

import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from backend.utils.helpers import observation_values
from configs.data import DataSettings


class V4AppTests(unittest.TestCase):
    def test_numeric_and_text_only_views_render(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        rows = [
            SimpleNamespace(
                test_name=name,
                observation_date=date,
                **observation_values({"value": value, "unit": "mg/L"}, DataSettings().common_inequalities),
            )
            for name, value, date in (
                ("Numeric", "4.50", datetime(2025, 1, 1)),
                ("Qualitative", "negative", datetime(2025, 1, 1)),
                ("Range", "0-1", None),
            )
        ]
        processor = SimpleNamespace(
            database=SimpleNamespace(
                get_patient=AsyncMock(return_value=1),
                get_patient_observations=AsyncMock(return_value=rows),
            )
        )
        with patch("backend.data_processing.laboratory_test.LaboratoryTestProcessor", return_value=processor):
            app = AppTest.from_file(Path(__file__).resolve().parents[2] / "src/app.py", default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            app.radio[0].set_value("Lab Test Dynamics").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.dataframe[0].value), 3)
            app.multiselect[0].set_value(["Numeric"]).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            self.assertEqual(app.metric[1].value, "0")
            self.assertEqual(app.dataframe[0].value.iloc[0]["Result"], "4.50")
            app.multiselect[0].set_value(["Qualitative", "Range"]).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("plotly_chart")), 0)
            self.assertEqual(set(app.dataframe[0].value["Result"]), {"negative", "0-1"})
            for row in rows:
                row.observation_date = None
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.date_input), 0)
            self.assertEqual(len(app.get("plotly_chart")), 0)


if __name__ == "__main__":
    unittest.main()
