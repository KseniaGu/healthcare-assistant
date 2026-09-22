"""Preserve original observation values alongside numeric projections.

Revision ID: c84d120a6e31
Revises: 90ba650e5a6c
"""

from alembic import op

revision = "c84d120a6e31"
down_revision = "90ba650e5a6c"
branch_labels = None
depends_on = None


def upgrade():
    # DEV startup creates tables from current metadata before applying migrations.
    op.execute("ALTER TABLE test_observation ADD COLUMN IF NOT EXISTS observed_value_text TEXT")


def downgrade():
    op.drop_column("test_observation", "observed_value_text")
