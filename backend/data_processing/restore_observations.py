"""Restore observation text and missing rows from stored report JSON; never call an LLM."""

import argparse
import json
from collections import Counter

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from backend.utils.helpers import observation_values
from backend.utils.schemas import TestObservationSchema
from configs.constants import DEFAULT_LAB_TEST_NAME
from configs.data import DataSettings
from configs.database import PostgreSQLSettings
from src.databases.postgres.models import Report, TestCatalog, TestObservation


def restore_observations(session: Session) -> dict:
    """Reconcile unambiguous stored source rows, preserving existing numeric values and flags.

    Historical placeholder names include the report UUID and source-row index, so
    they can be matched without guessing. Ambiguous matches are reported and skipped.
    The caller commits or rolls back the entire operation.
    """
    session.execute(text("LOCK TABLE report, test_observation, test_catalog IN SHARE ROW EXCLUSIVE MODE"))
    counts = {"added": 0, "renamed": 0, "text_restored": 0, "units_restored": 0, "skipped": []}
    catalogs = {row.canonical_name: row for row in session.scalars(select(TestCatalog))}
    inequalities = DataSettings().common_inequalities
    for report in session.scalars(select(Report).order_by(Report.id)):
        results = report.raw_json.get("results", [])
        if not isinstance(results, list):
            counts["skipped"].append({"report_id": report.id, "reason": "results is not an array"})
            continue
        existing = list(session.scalars(select(TestObservation).where(TestObservation.report_id == report.id)))
        names = [item.get("name") or item.get("test_name") for item in results if isinstance(item, dict)]
        name_counts = Counter(name for name in names if isinstance(name, str))
        for index, result in enumerate(results):
            name = (result.get("name") or result.get("test_name")) if isinstance(result, dict) else None
            if not isinstance(name, str) or not name.strip():
                counts["skipped"].append({"report_id": report.id, "row": index + 1, "reason": "missing name"})
                continue
            if str(result.get("value", "")).strip().lower() in ("выполняется", "pending", "in progress"):
                counts["skipped"].append({"report_id": report.id, "row": index + 1, "reason": "pending result"})
                continue
            placeholder = f"{DEFAULT_LAB_TEST_NAME}_{report.uuid}_{index}"
            matches = [row for row in existing if row.test_name in (name, placeholder)]
            if name_counts[name] != 1 or len(matches) > 1:
                counts["skipped"].append({"report_id": report.id, "row": index + 1, "reason": "ambiguous source match"})
                continue
            values = observation_values(result, inequalities)
            catalog = catalogs.get(name)
            if catalog is None:
                catalog = TestCatalog(canonical_name=name, preferred_unit=result.get("unit_normalized"))
                session.add(catalog)
                session.flush()
                catalogs[name] = catalog
            if not matches:
                observation = TestObservationSchema(
                    report_id=report.id,
                    patient_id=report.patient_id,
                    test_catalog_id=catalog.id,
                    test_name=name,
                    observation_date=report.report_date,
                    created_at=report.processed_at,
                    **values,
                )
                session.add(TestObservation(**observation.model_dump()))
                counts["added"] += 1
            else:
                observation = matches[0]
                if observation.test_name == placeholder:
                    observation.test_name = name
                    observation.test_catalog_id = catalog.id
                    counts["renamed"] += 1
                if observation.observed_value_text is None and values["observed_value_text"] is not None:
                    observation.observed_value_text = values["observed_value_text"]
                    counts["text_restored"] += 1
                if observation.unit is None and values["unit"] is not None:
                    observation.unit = values["unit"]
                    counts["units_restored"] += 1
    session.flush()
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Commit changes; default is a rolled-back preview.")
    args = parser.parse_args(argv)
    engine = create_engine(PostgreSQLSettings().get_connection_url())
    try:
        with Session(engine) as session:
            changes = restore_observations(session)
            if args.apply:
                session.commit()
            else:
                session.rollback()
            print(json.dumps({"applied": args.apply, **changes}, ensure_ascii=False, indent=2))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
