"""Catalog reuse must not depend on normalized-unit metadata absent from v4."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from backend.utils.schemas import TestCatalogSchema
from src.databases.postgres.database import PostgreSQLDB


class V4CatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_name_is_reused_when_preferred_unit_changes_or_is_absent(self):
        database = PostgreSQLDB.__new__(PostgreSQLDB)
        existing = SimpleNamespace(id=7, canonical_name="Synthetic", preferred_unit="mg/L")
        database.select = AsyncMock(return_value=Mock(first=Mock(return_value=existing)))
        session = Mock()
        for unit in (None, "g/L"):
            with self.subTest(unit=unit):
                identifier = await PostgreSQLDB.get_test_catalog_id.__wrapped__(
                    database, TestCatalogSchema(canonical_name="Synthetic", preferred_unit=unit), session=session
                )
                self.assertEqual(identifier, 7)
                filters = database.select.call_args.args[1].model_dump(exclude_unset=True)
                self.assertEqual(filters, {"canonical_name": "Synthetic"})
                session.add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
