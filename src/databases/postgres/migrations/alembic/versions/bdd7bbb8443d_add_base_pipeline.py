"""add base pipeline

Revision ID: bdd7bbb8443d
Revises: 
Create Date: 2025-12-10 02:31:37.202095

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from configs.enums import TaskEnum, PipelineStatusEnum

# revision identifiers, used by Alembic.
revision = 'bdd7bbb8443d'
down_revision = None
branch_labels = None
depends_on = None

_SEED_NAME = f"base_pipeline__seed__{revision}"


def upgrade():
    conn = op.get_bind()

    exists = conn.execute(text("SELECT EXISTS (SELECT 1 FROM pipeline LIMIT 1)")).scalar()
    description = """
    The general base pipeline:
        1. Update files with laboratory results by masking personally identifiable information (PII).
        2. Parse masked results to extract the data in a format that is most suitable for further processing.
        3. Pass the extracted data through the medical data processor.
        4. Add processed data to the database for further aggregation.
    """
    if not exists:
        conn.execute(
            text(
                """
                INSERT INTO pipeline (name, description, version, task, status)
                VALUES (:name, :description, :version, :task, :status)
                """
            ),
            {
                "name": _SEED_NAME, "description": description, "version": "1",
                "task": TaskEnum.laboratory_results.name, "status": PipelineStatusEnum.active.name
            }
        )


def downgrade():
    conn = op.get_bind()
    conn.execute(text("DELETE FROM pipeline WHERE name = :name"), {"name": _SEED_NAME})
